"""
eval_harness.py
------------------------------------------------------------
Runs a set of test questions through a RAGChain and produces the
actual JPMorgan-relevant artifact for this project: a report with
real numbers on hallucination rate and numeric accuracy, not a
vague claim of "it works well."

This is what you show in an interview instead of describing the
project verbally — a table of real scores against a real (small,
honest) test set.
------------------------------------------------------------
"""

from dataclasses import dataclass, field
from typing import List, Callable

from ..rag.qa_chain import RAGChain
from .numeric_accuracy import check_numeric_accuracy, NumericAccuracyResult
from .hallucination_checker import check_hallucination_signals, HallucinationCheckResult


@dataclass
class EvalCase:
    question: str
    # Optional: if you know the correct answer, you can additionally
    # check whether the expected value appears in the model's answer.
    # Left None for cases where you just want groundedness checked.
    expected_value: str | None = None


@dataclass
class EvalCaseResult:
    question: str
    answer: str
    numeric: NumericAccuracyResult
    hallucination: HallucinationCheckResult
    expected_value_found: bool | None = None


@dataclass
class EvalReport:
    results: List[EvalCaseResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def avg_numeric_accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.numeric.accuracy for r in self.results) / self.n

    @property
    def refusal_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.hallucination.is_refusal) / self.n

    @property
    def low_overlap_rate(self) -> float:
        """Fraction of answers flagged as poorly grounded in the source —
        the closest thing this harness has to a 'hallucination rate.'"""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.hallucination.flagged_low_overlap) / self.n

    @property
    def ungrounded_number_rate(self) -> float:
        """Fraction of answers that stated at least one number NOT
        present anywhere in the retrieved source context."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.numeric.has_ungrounded_numbers) / self.n

    def print_summary(self):
        print("=" * 60)
        print(f"Evaluation report — {self.n} test cases")
        print("=" * 60)
        print(f"Avg numeric accuracy (grounded/total numbers): {self.avg_numeric_accuracy:.1%}")
        print(f"Ungrounded-number rate (>=1 unsupported number): {self.ungrounded_number_rate:.1%}")
        print(f"Low source-overlap rate (flagged, needs review): {self.low_overlap_rate:.1%}")
        print(f"Refusal rate: {self.refusal_rate:.1%}")
        print("=" * 60)

        flagged = [r for r in self.results if r.numeric.has_ungrounded_numbers or r.hallucination.flagged_low_overlap]
        if flagged:
            print(f"\n{len(flagged)} case(s) flagged for review:\n")
            for r in flagged:
                print(f"  Q: {r.question}")
                print(f"  A: {r.answer[:150]}")
                if r.numeric.has_ungrounded_numbers:
                    print(f"     Ungrounded numbers: {r.numeric.ungrounded_numbers}")
                if r.hallucination.flagged_low_overlap:
                    print(f"     Low source overlap: {r.hallucination.source_overlap:.2f}")
                print()


def run_eval(chain: RAGChain, cases: List[EvalCase]) -> EvalReport:
    report = EvalReport()

    for case in cases:
        result = chain.ask(case.question)
        source_context = "\n".join(s.chunk.text for s in result.sources)

        numeric = check_numeric_accuracy(result.answer, source_context)
        hallucination = check_hallucination_signals(result.answer, source_context)

        expected_found = None
        if case.expected_value is not None:
            expected_found = case.expected_value.replace(",", "") in result.answer.replace(",", "")

        report.results.append(
            EvalCaseResult(
                question=case.question,
                answer=result.answer,
                numeric=numeric,
                hallucination=hallucination,
                expected_value_found=expected_found,
            )
        )

    return report

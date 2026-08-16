"""
hallucination_checker.py
------------------------------------------------------------
Complements numeric_accuracy.py with checks that aren't about
numbers specifically:

  1. Unsupported refusal check: did the model claim "the context
     doesn't contain this information" when it actually does? (An
     overly cautious model that refuses to answer things it CAN
     answer is a real, measurable failure mode — not just "did it
     make things up," but "is it actually using what it was given.")

  2. Source-overlap score: how much of the answer's wording actually
     overlaps with the retrieved context, as a cheap, fast proxy for
     "is this grounded" — real semantic entailment checking would use
     another LLM call ("does this source support this claim? yes/no"),
     which is more accurate but costs money and latency per check.
     This word-overlap version is the free, always-available first pass;
     the LLM-judge version is the natural upgrade once you have API
     budget for evaluation (see module docstring note below).

UPGRADE PATH: for a more rigorous check, replace `source_overlap_score`
with an LLM-as-judge call: feed the judge the claim + the source chunk,
ask "is this claim fully supported by the source? yes/no/partially."
That's strictly more accurate but costs one extra LLM call per
evaluated answer — worth it once Phase 2's dataset is large enough
that manual spot-checking isn't feasible.
------------------------------------------------------------
"""

import re
from dataclasses import dataclass

REFUSAL_PATTERNS = [
    r"don'?t know",
    r"not (?:enough|sufficient) information",
    r"cannot (?:determine|answer|find)",
    r"context doesn'?t (?:contain|provide|include)",
    r"no information (?:is )?(?:provided|available)",
]

_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL_RE.search(answer))


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def source_overlap_score(answer: str, source_context: str) -> float:
    """
    Fraction of the answer's meaningful (non-trivial) words that also
    appear somewhere in the source context. Cheap proxy for groundedness
    — NOT a substitute for real entailment checking (see module docstring),
    but useful as a fast, free first-pass signal across many examples.
    """
    stopwords = {
        "the", "a", "an", "is", "was", "were", "of", "to", "in", "on", "for",
        "and", "or", "per", "this", "that", "it", "its", "as", "by", "with",
        "excerpt", "answer",
    }
    answer_words = _tokenize(answer) - stopwords
    source_words = _tokenize(source_context)

    if not answer_words:
        return 1.0  # nothing substantive to check

    overlap = answer_words & source_words
    return len(overlap) / len(answer_words)


@dataclass
class HallucinationCheckResult:
    is_refusal: bool
    source_overlap: float           # 0.0-1.0, higher = more grounded
    flagged_low_overlap: bool       # True if overlap is below threshold — worth a human look

    LOW_OVERLAP_THRESHOLD = 0.4


def check_hallucination_signals(answer: str, source_context: str) -> HallucinationCheckResult:
    overlap = source_overlap_score(answer, source_context)
    return HallucinationCheckResult(
        is_refusal=is_refusal(answer),
        source_overlap=overlap,
        flagged_low_overlap=overlap < HallucinationCheckResult.LOW_OVERLAP_THRESHOLD,
    )

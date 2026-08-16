"""
numeric_accuracy.py
------------------------------------------------------------
The single most important check for a financial LLM system: when
the model states a number, is that number actually present in the
source it cited?

This deliberately does NOT ask "is the number correct in the real
world" (that requires ground truth this system usually doesn't
have) — it asks the narrower, checkable question: "is this number
grounded in the retrieved context, or did the model invent/
misremember it?" That's precisely the RAG failure mode from
qa_chain.py's docstring: the model can still misread a number
that's sitting right in front of it. This module is how you catch
that happening, systematically, instead of spot-checking by hand.
------------------------------------------------------------
"""

import re
from dataclasses import dataclass
from typing import List


NUMBER_PATTERN = re.compile(r"\$?[\d,]+(?:\.\d+)?%?")

# Strip citation self-references ("Excerpt 1", "chunk #2", "source 3")
# before extracting numbers — these are pointers to WHERE a fact came
# from, not factual claims themselves, and counting them as "numbers
# in the answer" produces false positives: a fully-grounded answer
# that cites "Excerpt 1" would otherwise get flagged for the ungrounded
# number "1", which defeats the point of this check (caught by testing
# this module against a known-good answer before trusting it).
_CITATION_REF_PATTERN = re.compile(r"\b(?:excerpt|chunk|source)\s*#?\s*\d+\b", re.IGNORECASE)


def extract_numbers(text: str) -> List[str]:
    """Extract number-like tokens (currency, percentages, plain numbers),
    normalized by stripping commas so '1,234' and '1234' compare equal.
    Citation self-references ('Excerpt 1') are excluded — see note above."""
    text_without_citations = _CITATION_REF_PATTERN.sub("", text)
    raw = NUMBER_PATTERN.findall(text_without_citations)
    return [n.replace(",", "") for n in raw if any(c.isdigit() for c in n)]


@dataclass
class NumericAccuracyResult:
    answer_numbers: List[str]
    grounded_numbers: List[str]      # numbers in the answer that DO appear in the source context
    ungrounded_numbers: List[str]    # numbers in the answer that DON'T appear in the source context
    accuracy: float                  # grounded / total (1.0 if answer contains no numbers at all)

    @property
    def has_ungrounded_numbers(self) -> bool:
        return len(self.ungrounded_numbers) > 0


def check_numeric_accuracy(answer: str, source_context: str) -> NumericAccuracyResult:
    """
    Compares every number mentioned in `answer` against every number
    present in `source_context` (the concatenation of retrieved chunks).
    A number counts as "grounded" if the same normalized digit string
    appears anywhere in the source context.
    """
    answer_numbers = extract_numbers(answer)
    source_numbers = set(extract_numbers(source_context))

    grounded = [n for n in answer_numbers if n in source_numbers]
    ungrounded = [n for n in answer_numbers if n not in source_numbers]

    accuracy = 1.0 if not answer_numbers else len(grounded) / len(answer_numbers)

    return NumericAccuracyResult(
        answer_numbers=answer_numbers,
        grounded_numbers=grounded,
        ungrounded_numbers=ungrounded,
        accuracy=accuracy,
    )

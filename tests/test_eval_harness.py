"""
test_eval_harness.py
------------------------------------------------------------
Tests the evaluation harness itself — the most important tests in
this project, since a broken eval harness would silently give you
false confidence about the model's real behavior. Fully offline.

Run: python -m pytest tests/ -v
------------------------------------------------------------
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.numeric_accuracy import extract_numbers, check_numeric_accuracy
from src.eval.hallucination_checker import is_refusal, source_overlap_score, check_hallucination_signals
from src.finetune.dataset_builder import extract_numeric_facts, build_templated_examples
from src.pipeline.chunker import Chunk


def test_extract_numbers_finds_currency_and_percent():
    numbers = extract_numbers("Revenue was $394.2 billion, up 6% from last year.")
    assert "$394.2" in numbers, f"expected '$394.2' in {numbers}"
    assert "6%" in numbers
    print("PASS: extract_numbers finds currency and percentage values")


def test_extract_numbers_excludes_citation_references():
    """Regression test for the exact bug caught during manual testing:
    'per Excerpt 1' should NOT count '1' as a factual number claim."""
    numbers = extract_numbers("Revenue was $394.2 billion, per Excerpt 1.")
    assert "1" not in numbers, "citation reference number should be excluded"
    assert "$394.2" in numbers
    print("PASS: extract_numbers excludes 'Excerpt N' citation self-references")


def test_numeric_accuracy_perfect_for_grounded_answer():
    answer = "Total net revenue was $394.2 billion, an increase of 6%."
    source = "Our total net revenue for fiscal year 2025 was $394.2 billion, an increase of 6%."
    result = check_numeric_accuracy(answer, source)
    assert result.accuracy == 1.0, f"expected 1.0 accuracy, got {result.accuracy}"
    assert not result.has_ungrounded_numbers
    print("PASS: numeric accuracy is 1.0 when every stated number is in the source")


def test_numeric_accuracy_flags_invented_numbers():
    answer = "Total net revenue was $450.7 billion, an increase of 12%."  # both invented
    source = "Our total net revenue for fiscal year 2025 was $394.2 billion, an increase of 6%."
    result = check_numeric_accuracy(answer, source)
    assert result.accuracy == 0.0, f"expected 0.0 accuracy, got {result.accuracy}"
    assert result.has_ungrounded_numbers
    assert set(result.ungrounded_numbers) == {"$450.7", "12%"}
    print("PASS: numeric accuracy correctly flags fully invented numbers")


def test_numeric_accuracy_partial_grounding():
    answer = "Revenue was $394.2 billion, roughly a 15% increase."  # 394.2 real, 15% invented
    source = "Our total net revenue for fiscal year 2025 was $394.2 billion, an increase of 6%."
    result = check_numeric_accuracy(answer, source)
    assert 0.0 < result.accuracy < 1.0, f"expected partial accuracy, got {result.accuracy}"
    assert "15%" in result.ungrounded_numbers
    assert "$394.2" in result.grounded_numbers
    print(f"PASS: numeric accuracy correctly scores partial grounding ({result.accuracy:.0%})")


def test_is_refusal_detects_refusal_language():
    assert is_refusal("I don't know based on the provided context.")
    assert is_refusal("The context doesn't contain this information.")
    assert not is_refusal("Total net revenue was $394.2 billion.")
    print("PASS: is_refusal correctly detects refusal vs. substantive answers")


def test_source_overlap_high_for_grounded_answer():
    answer = "Total net revenue was $394.2 billion, an increase of 6%."
    source = "Our total net revenue for fiscal year 2025 was $394.2 billion, an increase of 6% compared to the prior year."
    score = source_overlap_score(answer, source)
    assert score > 0.7, f"expected high overlap for a grounded answer, got {score}"
    print(f"PASS: source_overlap_score is high ({score:.2f}) for a grounded answer")


def test_source_overlap_low_for_unrelated_answer():
    answer = "The company plans to launch a new product line in emerging markets next year."
    source = "Our total net revenue for fiscal year 2025 was $394.2 billion, an increase of 6%."
    score = source_overlap_score(answer, source)
    assert score < 0.4, f"expected low overlap for an unrelated answer, got {score}"
    print(f"PASS: source_overlap_score is low ({score:.2f}) for an unrelated/unsupported answer")


def test_hallucination_signals_end_to_end():
    good = check_hallucination_signals(
        "Total net revenue was $394.2 billion, an increase of 6%.",
        "Our total net revenue for fiscal year 2025 was $394.2 billion, an increase of 6%.",
    )
    assert not good.flagged_low_overlap

    bad = check_hallucination_signals(
        "The company is planning a merger with a major competitor next quarter.",
        "Our total net revenue for fiscal year 2025 was $394.2 billion, an increase of 6%.",
    )
    assert bad.flagged_low_overlap
    print("PASS: check_hallucination_signals flags unsupported claims, passes grounded ones")


def test_dataset_builder_excludes_garbage_labels():
    """Regression test for the exact bug caught during manual testing:
    'What was an increase?' / 'What was which?' should never be generated —
    while still keeping legitimate facts from the same text (not just
    filtering everything out, which would make this test trivially pass)."""
    text = (
        "Gross margin was 46.2%. Stockholders equity was $145.3 billion, "
        "an increase of 6% compared to the prior year, which grew steadily."
    )
    facts = extract_numeric_facts(text)
    labels = [f["label"].lower() for f in facts]

    assert not any(label.startswith(("an ", "which", "a ", "the ")) for label in labels), (
        f"found a garbage label in {labels}"
    )
    assert any("gross margin" in label or "stockholders equity" in label for label in labels), (
        f"expected at least one legitimate fact to survive filtering, got {labels}"
    )
    print(f"PASS: dataset builder keeps legitimate facts while excluding garbage labels ({labels})")


def test_build_templated_examples_produces_valid_questions():
    chunk = Chunk(
        text="Stockholders' equity was $145.3 billion. Total liabilities were $267.3 billion.",
        company="AAPL", filing_type="10-K", source_id="t1", chunk_index=0,
    )
    examples = build_templated_examples([chunk])
    assert len(examples) >= 2
    for ex in examples:
        assert ex.question.startswith("What was")
        assert "?" in ex.question
        assert len(ex.answer) > 0
    print(f"PASS: build_templated_examples produced {len(examples)} well-formed Q&A pairs")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} tests...\n")
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {test.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    if failures:
        sys.exit(1)

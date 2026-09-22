import pytest

from eval.metrics import (
    answer_contains_expected_facts,
    is_refusal,
    reciprocal_rank,
    recall_at_k,
)


class TestRecallAtK:

    def test_hit_in_top_k(self):
        texts = ["irrelevant", "mentions Python here", "also irrelevant"]
        assert recall_at_k(texts, ["Python"], k=3) == 1.0

    def test_hit_outside_k_counts_as_miss(self):
        texts = ["irrelevant", "irrelevant", "mentions Python here"]
        assert recall_at_k(texts, ["Python"], k=2) == 0.0

    def test_no_hit_is_zero(self):
        texts = ["irrelevant", "also irrelevant"]
        assert recall_at_k(texts, ["Python"], k=2) == 0.0

    def test_case_insensitive(self):
        texts = ["mentions PYTHON here"]
        assert recall_at_k(texts, ["python"], k=1) == 1.0

    def test_empty_keywords_raises(self):
        with pytest.raises(ValueError):
            recall_at_k(["text"], [], k=1)


class TestReciprocalRank:

    def test_first_position_is_rank_one(self):
        texts = ["mentions Python", "irrelevant"]
        assert reciprocal_rank(texts, ["Python"]) == 1.0

    def test_third_position_is_one_third(self):
        texts = ["irrelevant", "irrelevant", "mentions Python"]
        assert reciprocal_rank(texts, ["Python"]) == pytest.approx(1 / 3)

    def test_no_match_is_zero(self):
        texts = ["irrelevant", "irrelevant"]
        assert reciprocal_rank(texts, ["Python"]) == 0.0


class TestAnswerContainsExpectedFacts:

    def test_all_keywords_present(self):
        answer = "This person knows Python and SQL."
        assert answer_contains_expected_facts(
            answer, ["Python", "SQL"], min_fraction=1.0
        )

    def test_partial_keywords_fail_at_full_fraction(self):
        answer = "This person knows Python."
        assert not answer_contains_expected_facts(
            answer, ["Python", "SQL"], min_fraction=1.0
        )

    def test_partial_keywords_pass_at_half_fraction(self):
        answer = "This person knows Python."
        assert answer_contains_expected_facts(
            answer, ["Python", "SQL"], min_fraction=0.5
        )


class TestIsRefusal:

    def test_detects_refusal_marker(self):
        marker = "I couldn't find that information in the uploaded documents."
        assert is_refusal(
            "I couldn't find that information in the uploaded documents.",
            marker,
        )

    def test_normal_answer_is_not_a_refusal(self):
        marker = "I couldn't find that information in the uploaded documents."
        assert not is_refusal("The answer is 42.", marker)

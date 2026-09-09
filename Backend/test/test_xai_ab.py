"""Comparing two benchmark runs.

The arithmetic is easy and the restraint is not. A tool that turns a two-
question difference on forty questions into "the schema change hurt accuracy"
is worse than no tool, because the number looks like evidence. So most of what
is tested here is the refusal: that small differences come back as consistent
with noise, and that mismatched question sets are noticed rather than compared
row by row.
"""

import json
import sys
from pathlib import Path

import pytest

# Backend/test/ -> repo root -> benchmark/xai_eval
_TOOL_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "xai_eval"
sys.path.insert(0, str(_TOOL_DIR))

from ab_rationale import compare, load_runs, mcnemar  # noqa: E402


def _question(text: str, correct: bool) -> dict:
    return {
        "question": text,
        "tool_matches": {"missed": [] if correct else ["SomeAgent--SomeAction"],
                         "extra": []},
    }


def _run(outcomes: dict[str, bool], tokens: int = 10_000) -> dict:
    return {
        "method": "tool-llm",
        "model": "openai/gpt-4o-mini",
        "summary": {"total_token_usage": tokens, "total_server_time": 100.0},
        "questions": [_question(q, ok) for q, ok in outcomes.items()],
    }


class TestMcnemar:
    def test_no_change_is_no_evidence(self):
        assert mcnemar(both_ways=40, only_a=0, only_b=0) == 1.0

    def test_a_symmetric_split_is_no_evidence(self):
        """Three fixed and three broken is the schema change doing nothing."""
        assert mcnemar(both_ways=30, only_a=3, only_b=3) == 1.0

    def test_a_small_one_sided_difference_is_not_significant(self):
        """Two questions moving one way is what a re-run produces."""
        assert mcnemar(both_ways=38, only_a=0, only_b=2) > 0.05

    def test_a_large_one_sided_difference_is(self):
        assert mcnemar(both_ways=30, only_a=0, only_b=10) < 0.05

    def test_only_the_changed_questions_count(self):
        """A question both runs got right says nothing about the change."""
        assert mcnemar(both_ways=5, only_a=1, only_b=4) == mcnemar(
            both_ways=500, only_a=1, only_b=4)

    @pytest.mark.parametrize("a,b", [(0, 1), (1, 0), (5, 5), (0, 20)])
    def test_it_stays_a_probability(self, a, b):
        assert 0.0 <= mcnemar(10, a, b) <= 1.0


class TestPairing:
    def test_questions_are_matched_by_text_not_position(self):
        """A run that skipped or reordered questions would otherwise be
        compared row by row and silently produce nonsense."""
        with_ = _run({"a": True, "b": False, "c": True})
        without = _run({"c": True, "a": True, "b": False})

        result = compare(with_, without)

        assert result["shared_questions"] == 3
        assert result["fixed_by_rationale"] == 0
        assert result["broken_by_rationale"] == 0

    def test_questions_missing_from_one_run_are_reported_not_compared(self):
        with_ = _run({"a": True, "b": True})
        without = _run({"a": True, "c": True})

        result = compare(with_, without)

        assert result["shared_questions"] == 1
        assert result["unmatched_questions"] == 2

    def test_a_fix_and_a_break_are_counted_separately(self):
        """Direction matters: three each way is noise, three one way is not."""
        with_ = _run({"a": True, "b": False, "c": True})
        without = _run({"a": False, "b": True, "c": True})

        result = compare(with_, without)

        assert result["fixed_by_rationale"] == 1
        assert result["broken_by_rationale"] == 1
        assert result["p_value"] == 1.0

    def test_identical_runs_report_no_change(self):
        outcomes = {"a": True, "b": False, "c": True}

        result = compare(_run(outcomes), _run(outcomes))

        assert result["changed_questions"] == []
        assert result["p_value"] == 1.0


class TestWhatIsReported:
    def test_the_changed_questions_are_named(self):
        """A count says something moved; the list says what to go and look at."""
        with_ = _run({"a": True, "b": True})
        without = _run({"a": True, "b": False})

        changed = compare(with_, without)["changed_questions"]

        assert [c["question"] for c in changed] == ["b"]
        assert changed[0]["with_rationale"] is True

    def test_token_totals_are_carried_for_the_overhead(self):
        result = compare(_run({"a": True}, tokens=10_500),
                         _run({"a": True}, tokens=10_000))

        assert result["tokens_with"] - result["tokens_without"] == 500


class TestLoading:
    def test_a_results_file_flattens_to_one_entry_per_combination(self, tmp_path):
        path = tmp_path / "results.json"
        path.write_text(json.dumps({
            "tool-llm": {
                "openai/gpt-4o-mini": {"summary": {}, "questions": []},
                "openai/gpt-4o": {"summary": {}, "questions": []},
            },
            "simple": {"openai/gpt-4o-mini": {"summary": {}, "questions": []}},
        }))

        runs = load_runs(path)

        assert len(runs) == 3
        assert {r["method"] for r in runs} == {"tool-llm", "simple"}
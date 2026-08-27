"""Telling a failed tool call apart from a successful one.

Errors are written into `result` as prose by four different paths in
invoke_tool. Everything downstream -- the confidence signals, the "why not"
answers, the retry markers in the chain view -- needs to know which results are
failures, and none of it should be re-reading that prose to find out.

The risk this guards against is over-eager matching: a tool whose job is to
report on errors returns text full of the word, and marking those calls failed
would make the whole signal useless.
"""

import pytest

from src.tool_calling import _classify_result


class TestFailuresAreDetected:
    @pytest.mark.parametrize("result", [
        "Failed to invoke tool.\nExecution denied by user settings, do not attempt again.",
        "Failed to invoke MCP tool.\nError: connection refused",
        "Failed to invoke Internal tool.\nCause: KeyError('room')",
        "Failed to invoke OPACA tool.\nStatus code: 500\nResponse: boom",
    ])
    def test_every_failure_path_is_recognised(self, result):
        """One case per branch in invoke_tool that writes an error."""
        success, _ = _classify_result(result)

        assert success is False

    def test_the_cause_is_extracted_not_the_boilerplate(self):
        _, error = _classify_result("Failed to invoke MCP tool.\nError: connection refused")

        assert error == "Error: connection refused"

    def test_a_failure_with_no_detail_still_yields_something(self):
        """An empty error field would read as "failed, no idea why"."""
        _, error = _classify_result("Failed to invoke tool.")

        assert error


class TestSuccessesAreNotMisread:
    def test_a_result_about_errors_is_still_a_success(self):
        """The case that makes prefix matching necessary rather than substring."""
        success, error = _classify_result(
            "Scanned 400 lines. Failed to invoke tool. appears 3 times in the log."
        )

        assert success is True
        assert error is None

    @pytest.mark.parametrize("result", [
        "Room C.12 is free from 14:00.",
        {"rooms": ["C.12"], "count": 1},
        ["C.12", "D.03"],
        None,
        0,
        "",
    ])
    def test_ordinary_results_pass_through(self, result):
        """Tool results are not always strings, and an empty one is not an error."""
        success, error = _classify_result(result)

        assert success is True
        assert error is None
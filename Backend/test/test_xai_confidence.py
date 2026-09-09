"""Judging a chain from the record rather than by asking the model.

The rules are ceilings, not penalties: an answer built on a failed call is not
a high-confidence answer whatever else went right. Scores that add up invite
tuning until they produce the wanted number.

What is not tested here, because it is not claimed: that the level tracks
whether the answer is true. It tracks whether the chain ran cleanly, converged
and grounded its arguments.
"""

import pytest

from src.models import AgentMessage, QueryResponse, StepType, ToolCall, ToolType
from src.xai.confidence import assess, lower_only
from src.xai.provenance import build_edges
from src.xai.trace import build_trace


def _report(response: QueryResponse):
    trace = build_trace(response)
    return assess(trace, build_edges(trace))


def _call(call_id="m1/0", name="RoomAgent--GetRoomId", **kw) -> ToolCall:
    return ToolCall(id=call_id, type=ToolType.OPACA, name=name, **kw)


def _clean() -> QueryResponse:
    """A lookup that worked, with its argument taken from the question."""
    return QueryResponse(query="CO2 in the Experience Hub?", agent_messages=[
        AgentMessage(agent="Tool Generator", step_type=StepType.TOOL_CALL, tools=[
            _call(args={"room_name": "Experience Hub"}, result=1, success=True),
        ]),
        AgentMessage(agent="Tool Evaluator", step_type=StepType.EVALUATE,
                     formatted_output={"decision": "FINISHED"}),
        AgentMessage(agent="Output Generator", step_type=StepType.OUTPUT),
    ], content="600 ppm")


class TestACleanChain:
    def test_nothing_to_report_means_high(self):
        report = _report(_clean())

        assert report.level == "high"
        assert report.is_clean

    def test_a_chain_with_no_tools_is_not_penalised_by_itself(self):
        """Plenty of questions need no tool. Silence is not a finding."""
        report = _report(QueryResponse(query="hello", agent_messages=[
            AgentMessage(agent="assistant", step_type=StepType.OUTPUT)
        ], content="hi"))

        assert report.level == "high"


class TestCeilings:
    def test_a_failed_call_caps_at_medium(self):
        response = _clean()
        response.agent_messages[0].tools[0].success = False
        response.agent_messages[0].tools[0].error = "timeout"

        report = _report(response)

        assert report.level == "medium"
        assert any(s.id == "failed_calls" for s in report.signals)

    def test_a_regenerated_call_caps_at_medium(self):
        response = _clean()
        response.agent_messages.insert(1, AgentMessage(
            agent="Tool Generator", step_type=StepType.CORRECTION))

        report = _report(response)

        assert report.level == "medium"
        assert any(s.id == "corrections" for s in report.signals)

    def test_an_argument_from_nowhere_caps_at_medium(self):
        response = _clean()
        response.agent_messages[0].tools[0].args = {"organizer": "Dr. Weber"}

        report = _report(response)

        assert report.level == "medium"
        assert "organizer" in next(
            s.detail for s in report.signals if s.id == "untraceable_arguments")

    def test_stopping_while_the_evaluator_wanted_more_caps_at_low(self):
        """The clearest evidence in the trace that an answer was cut short."""
        response = _clean()
        response.agent_messages[1].formatted_output = {"decision": "CONTINUE"}

        report = _report(response)

        assert report.level == "low"
        assert any(s.id == "not_converged" for s in report.signals)

    def test_an_errored_request_caps_at_low(self):
        response = _clean()
        response.error = "OPACA unreachable"

        assert _report(response).level == "low"

    def test_the_lowest_ceiling_wins(self):
        response = _clean()
        response.agent_messages[0].tools[0].success = False       # medium
        response.agent_messages[1].formatted_output = {"decision": "CONTINUE"}  # low

        report = _report(response)

        assert report.level == "low"
        assert len(report.signals) == 2

    def test_only_the_last_evaluation_counts(self):
        """An early CONTINUE is the loop working, not a chain cut short."""
        response = _clean()
        response.agent_messages.insert(1, AgentMessage(
            agent="Tool Evaluator", step_type=StepType.EVALUATE,
            formatted_output={"decision": "CONTINUE"}))

        assert _report(response).level == "high"


class TestSignalsAreActionable:
    def test_a_failure_names_the_action(self):
        response = _clean()
        response.agent_messages[0].tools[0].success = False

        detail = next(s.detail for s in _report(response).signals if s.id == "failed_calls")

        assert "GetRoomId" in detail

    def test_a_weak_argument_is_not_reported_as_supplied(self):
        """Too common to judge is not the same as came from nowhere."""
        response = _clean()
        response.agent_messages[0].tools[0].args = {"limit": 1}

        assert _report(response).is_clean


class TestTheModelMayOnlyLower:
    @pytest.mark.parametrize("computed,proposed,expected", [
        ("high", "low", "low"),
        ("high", "medium", "medium"),
        ("medium", "low", "low"),
        ("medium", "high", "medium"),
        ("low", "high", "low"),
        ("low", "medium", "low"),
    ])
    def test_a_proposal_can_lower_but_never_raise(self, computed, proposed, expected):
        """Raising would let the assistant overrule the record about itself,
        which is the failure the computed level exists to prevent."""
        assert lower_only(computed, proposed) == expected

    @pytest.mark.parametrize("proposed", [None, "", "very high", "HIGH"])
    def test_an_unusable_proposal_leaves_the_level_alone(self, proposed):
        assert lower_only("medium", proposed) == "medium"
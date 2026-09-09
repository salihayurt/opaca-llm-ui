"""Normalising a QueryResponse into an ExecutionTrace.

Three things are worth failing a build over: that the shape is the same
whichever method produced it, that a large tool result cannot blow up a prompt,
and that credentials do not survive the trip.
"""

import pytest

from src.models import AgentMessage, QueryResponse, StepType, ToolCall, ToolType
from src.xai.trace import build_trace, digest, redact_args


def _call(call_id="m1/0", name="RoomAgent--GetFreeRooms", **kw) -> ToolCall:
    return ToolCall(id=call_id, type=ToolType.OPACA, name=name, **kw)


def _response(*messages, query="Which rooms are free?", content="answer") -> QueryResponse:
    return QueryResponse(query=query, agent_messages=list(messages), content=content)


class TestShapeIsMethodIndependent:
    def test_a_single_step_method_produces_one_step(self):
        """simple / simple-tools: one agent, one message per round."""
        trace = build_trace(_response(
            AgentMessage(agent="assistant", step_type=StepType.OUTPUT, model="openai/gpt-4o-mini")
        ))

        assert len(trace.steps) == 1
        assert trace.steps[0].step_type is StepType.OUTPUT

    def test_a_staged_method_produces_typed_steps(self):
        """tool-llm: generator, evaluator, output."""
        trace = build_trace(_response(
            AgentMessage(agent="Tool Generator", step_type=StepType.TOOL_CALL, tools=[_call()]),
            AgentMessage(agent="Tool Evaluator", step_type=StepType.EVALUATE),
            AgentMessage(agent="Output Generator", step_type=StepType.OUTPUT),
        ))

        assert [s.step_type for s in trace.steps] == [
            StepType.TOOL_CALL, StepType.EVALUATE, StepType.OUTPUT
        ]
        assert len(trace.steps_of(StepType.EVALUATE)) == 1

    def test_nesting_is_preserved(self):
        """self-orchestrated: workers point back at the step that spawned them."""
        trace = build_trace(_response(
            AgentMessage(id="orch", agent="Orchestrator", step_type=StepType.ROUTING),
            AgentMessage(id="w1", agent="WorkerAgent", parent_id="orch", iteration=1),
            AgentMessage(id="w2", agent="WorkerAgent", parent_id="orch", iteration=1),
        ))

        workers = [s for s in trace.steps if s.parent_id == "orch"]
        assert len(workers) == 2
        assert all(w.iteration == 1 for w in workers)

    def test_calls_are_lifted_and_ordered(self):
        """Provenance asks about the call sequence, not the step tree."""
        trace = build_trace(_response(
            AgentMessage(agent="a", tools=[_call("m1/0"), _call("m1/1", name="A--b")]),
            AgentMessage(agent="b", tools=[_call("m2/0", name="A--c")]),
        ))

        assert [c.order for c in trace.calls] == [0, 1, 2]
        assert [c.id for c in trace.calls] == ["m1/0", "m1/1", "m2/0"]
        assert trace.calls[2].step_id == trace.steps[1].id

    def test_call_names_are_split(self):
        trace = build_trace(_response(AgentMessage(agent="a", tools=[_call()])))

        assert trace.calls[0].agent == "RoomAgent"
        assert trace.calls[0].action == "GetFreeRooms"

    def test_an_empty_response_is_a_valid_trace(self):
        """Aborted or failed generations must not crash the reader."""
        trace = build_trace(QueryResponse(query="q"))

        assert trace.steps == []
        assert trace.calls == []


class TestDigest:
    def test_short_values_are_untouched(self):
        assert digest("Room C.12 is free") == "Room C.12 is free"
        assert digest(42) == 42

    def test_long_strings_are_truncated_and_say_so(self):
        result = digest("x" * 5000)

        assert len(result) < 400
        assert "+4700 chars" in result

    def test_lists_keep_their_length(self):
        """That there were eight rooms is the point; the eight objects are not."""
        result = digest([{"room": f"C.{i}"} for i in range(8)])

        assert len(result) == 3
        assert "+6 more items" in result[-1]

    def test_wide_dicts_are_capped(self):
        result = digest({f"key{i}": i for i in range(30)})

        assert len(result) <= 13
        assert "more keys" in str(result)

    def test_recursion_is_bounded(self):
        """A self-similar result must not recurse until the stack dies."""
        deep = {"a": {"b": {"c": {"d": {"e": {"f": "bottom"}}}}}}

        assert "[…]" in str(digest(deep))

    def test_shape_survives(self):
        result = digest({"rooms": ["C.12", "D.03"], "count": 2})

        assert isinstance(result, dict)
        assert isinstance(result["rooms"], list)
        assert result["count"] == 2


class TestRedaction:
    """The trace reaches both the user and, later, an explainer model.

    Container login sends credentials through tool arguments, so this is the
    one place that has to hold.
    """

    @pytest.mark.parametrize("key", [
        "password", "user_password", "apiKey", "api_key", "secret",
        "token", "access_token", "Authorization", "private_key",
    ])
    def test_secret_arguments_are_dropped(self, key):
        assert redact_args({key: "hunter2"})[key] == "[redacted]"

    def test_ordinary_arguments_are_kept(self):
        args = redact_args({"room": "C.12", "date": "2026-08-27", "attendees": 8})

        assert args == {"room": "C.12", "date": "2026-08-27", "attendees": 8}

    def test_secrets_nested_in_results_are_dropped(self):
        result = digest({"session": {"user": "salih", "token": "abc123"}})

        assert result["session"]["token"] == "[redacted]"
        assert result["session"]["user"] == "salih"


class TestDerivedViews:
    def test_failed_calls_are_collected(self):
        trace = build_trace(_response(AgentMessage(agent="a", tools=[
            _call("m1/0", success=True),
            _call("m1/1", name="A--b", success=False, error="timeout"),
        ])))

        assert [c.id for c in trace.failed_calls] == ["m1/1"]

    def test_models_are_deduplicated(self):
        trace = build_trace(_response(
            AgentMessage(agent="a", model="openai/gpt-4o-mini"),
            AgentMessage(agent="b", model="openai/gpt-4o-mini"),
            AgentMessage(agent="c", model="openai/gpt-4o"),
        ))

        assert trace.models_used == ["openai/gpt-4o", "openai/gpt-4o-mini"]

    def test_content_is_collapsed(self):
        trace = build_trace(_response(
            AgentMessage(agent="a", content="line\n\n   spaced    out\ttext")
        ))

        assert trace.steps[0].content == "line spaced out text"


class TestHash:
    def test_the_same_response_hashes_the_same(self):
        response = _response(AgentMessage(id="m1", agent="a", tools=[_call()]))

        assert build_trace(response).hash == build_trace(response).hash

    def test_a_different_call_changes_the_hash(self):
        base = _response(AgentMessage(id="m1", agent="a", tools=[_call()]))
        other = _response(AgentMessage(id="m1", agent="a", tools=[
            _call(name="RoomAgent--BookRoom")
        ]))

        assert build_trace(base).hash != build_trace(other).hash

    def test_timing_does_not_change_the_hash(self):
        """Timings differ between a live object and the same one from MongoDB.

        Including them would invalidate every cached explanation on restart.
        """
        fast = _response(AgentMessage(id="m1", agent="a", execution_time=0.5))
        slow = _response(AgentMessage(id="m1", agent="a", execution_time=9.9))

        assert build_trace(fast).hash == build_trace(slow).hash
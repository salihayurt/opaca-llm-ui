"""Trace fields the explanation layer reads.

These fields were added to models that are persisted to MongoDB and sent to the
frontend, so the tests here are mostly about not breaking what already works:
old documents must still load, and what the methods hand to the LLM must not
change. The one behavioural claim is that a failed tool call is distinguishable
from a successful one without parsing prose.
"""

import json

import pytest

from src.models import AgentMessage, QueryResponse, StepType, ToolCall, ToolType


def _tool(**overrides) -> ToolCall:
    args = {"id": "m1/0", "type": ToolType.OPACA, "name": "RoomAgent--GetFreeRooms"}
    args.update(overrides)
    return ToolCall(**args)


class TestAgentMessageDefaults:
    """A step that sets nothing must still be a valid step."""

    def test_defaults_are_complete(self):
        message = AgentMessage(agent="assistant")

        assert message.step_type is StepType.TOOL_CALL
        assert message.model == ""
        assert message.iteration == 0
        assert message.parent_id is None

    def test_old_documents_still_load(self):
        """Sessions stored before these fields existed are read back on startup."""
        stored = {
            "id": "abc",
            "agent": "Tool Generator",
            "content": "…",
            "tools": [],
            "response_metadata": {"input_tokens": 10},
            "execution_time": 1.5,
            "formatted_output": None,
        }

        message = AgentMessage.model_validate(stored)

        assert message.agent == "Tool Generator"
        assert message.step_type is StepType.TOOL_CALL

    def test_step_type_survives_a_round_trip(self):
        """Explanations are rebuilt from stored chats, not only from live ones."""
        message = AgentMessage(
            agent="Orchestrator", step_type=StepType.ROUTING, model="openai/gpt-4o-mini",
            iteration=2, parent_id="parent-1",
        )

        restored = AgentMessage.model_validate_json(message.model_dump_json())

        assert restored.step_type is StepType.ROUTING
        assert restored.model == "openai/gpt-4o-mini"
        assert restored.iteration == 2
        assert restored.parent_id == "parent-1"

    def test_step_type_serialises_as_its_value(self):
        """The frontend reads JSON, so the wire form has to be the plain string."""
        payload = json.loads(AgentMessage(agent="a", step_type=StepType.EVALUATE).model_dump_json())

        assert payload["step_type"] == "evaluate"

    def test_unknown_step_type_is_rejected(self):
        with pytest.raises(Exception):
            AgentMessage(agent="a", step_type="something-else")


class TestToolCallOutcome:
    def test_outcome_is_unset_before_invocation(self):
        """None is not False: a call awaiting approval has not failed."""
        assert _tool().success is None

    def test_failure_is_distinguishable_from_a_result_that_reads_like_one(self):
        """The point of the field. Both results mention an error; only one is one."""
        failed = _tool(result="Failed to invoke tool.\nCause: timeout", success=False,
                       error="timeout")
        succeeded = _tool(id="m1/1", result="No errors were found in the log.", success=True)

        assert failed.success is False
        assert succeeded.success is True

    def test_old_tool_calls_still_load(self):
        stored = {"id": "m1/0", "type": "opaca", "name": "A--b", "args": {}, "result": "ok"}

        assert ToolCall.model_validate(stored).success is None


class TestWhatTheModelSees:
    """without_id() feeds tool history back to the LLM.

    Anything added here changes the prompt, and a change in the prompt is a
    change in how the methods behave. That is a task-solving change wearing an
    explainability costume, so the new fields stay out.
    """

    def test_bookkeeping_fields_are_hidden(self):
        call = _tool(result="Room C.12", success=True, error=None)

        assert set(call.without_id()) == {"type", "name", "args", "result"}

    def test_error_text_is_hidden_too(self):
        """The failure is already visible in `result`; repeating it would be new input."""
        call = _tool(result="Failed to invoke tool.\nCause: timeout", success=False,
                     error="timeout")

        assert "error" not in call.without_id()
        assert "timeout" in call.without_id()["result"]


class TestResponseIdentity:
    def test_each_response_gets_its_own_id(self):
        assert QueryResponse(query="a").response_id != QueryResponse(query="b").response_id

    def test_id_is_stable_across_storage(self):
        response = QueryResponse(query="Which rooms are free?")

        restored = QueryResponse.model_validate_json(response.model_dump_json())

        assert restored.response_id == response.response_id

    def test_old_responses_are_given_an_id_on_load(self):
        """Chats stored before this field existed must not fail to load."""
        stored = {"query": "q", "agent_messages": [], "iterations": 1,
                  "execution_time": 2.0, "content": "answer", "error": ""}

        assert QueryResponse.model_validate(stored).response_id
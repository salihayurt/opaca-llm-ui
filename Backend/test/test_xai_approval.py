"""What the user is shown before allowing a tool call.

This is the only point where an explanation supports a decision rather than
describing one: the call has not run, and the user can still stop it. So the
context has to be established fact, and it must never be the reason a call
fails -- a prompt that errors out blocks a tool the user was willing to allow.
"""

import pytest

from src.models import (
    AgentMessage, ConfirmActionNotification, QueryResponse, ToolCall, ToolType,
)
from src.xai.provenance import edges_for_call
from src.xai.trace import build_trace


def _tool(name="RoomAgent--BookRoom", args=None, call_id="m2/0") -> ToolCall:
    return ToolCall(id=call_id, type=ToolType.OPACA, name=name, args=args or {})


def _history(query="Book a room", **kw) -> QueryResponse:
    """A response with one completed lookup already in it."""
    return QueryResponse(query=query, agent_messages=[
        AgentMessage(agent="Tool Generator", tools=[ToolCall(
            id="m1/0", type=ToolType.OPACA, name="RoomAgent--GetFreeRooms",
            args={}, result={"rooms": [{"id": "C.12", "seats": 8}]}, success=True,
        )]),
    ], **kw)


class TestPendingCallResolution:
    def test_an_argument_from_an_earlier_result_is_traced(self):
        trace = build_trace(_history())

        edges = edges_for_call("m2/0", {"room": "C.12"}, trace)

        assert edges[0].source == "tool_result"
        assert edges[0].from_call_name == "RoomAgent--GetFreeRooms"

    def test_an_argument_from_nowhere_is_flagged(self):
        """The line that should change someone's mind."""
        trace = build_trace(_history())

        edges = edges_for_call("m2/0", {"organizer": "Dr. Weber"}, trace)

        assert edges[0].source == "unmatched"

    def test_it_works_when_the_pending_call_is_already_in_the_trace(self):
        """tool-llm appends the step before invoking; the call must not match itself."""
        response = _history()
        response.agent_messages.append(AgentMessage(
            agent="Tool Generator", tools=[_tool(args={"room": "C.12"})]
        ))
        trace = build_trace(response)

        edges = edges_for_call("m2/0", {"room": "C.12"}, trace)

        assert edges[0].from_call_name == "RoomAgent--GetFreeRooms"

    def test_it_works_when_the_pending_call_is_not_in_the_trace_yet(self):
        """self-orchestrated appends after invoking."""
        trace = build_trace(_history())

        edges = edges_for_call("unknown-id", {"room": "C.12"}, trace)

        assert edges[0].source == "tool_result"

    def test_a_sibling_call_awaiting_its_own_result_finds_nothing(self):
        """Batched calls have no results yet; searching them must not invent one."""
        response = _history()
        response.agent_messages.append(AgentMessage(agent="g", tools=[
            ToolCall(id="m2/0", type=ToolType.OPACA, name="A--x", args={"code": "Zeta9981"}),
            ToolCall(id="m2/1", type=ToolType.OPACA, name="A--y", args={}),
        ]))
        trace = build_trace(response)

        edges = edges_for_call("m2/1", {"code": "Zeta9981"}, trace)

        assert edges[0].source != "tool_result"

    def test_a_call_without_arguments_produces_nothing_to_show(self):
        assert edges_for_call("m2/0", {}, build_trace(_history())) == []


class TestNotificationShape:
    def test_the_old_fields_still_work_without_the_new_ones(self):
        """The dialog must render for a ToolCaller built without a response."""
        notification = ConfirmActionNotification(tool="A--b", params={"x": 1})

        assert notification.sources == []
        assert notification.prior_calls == 0

    def test_it_serialises_for_the_websocket(self):
        from src.models import ParamSource

        notification = ConfirmActionNotification(
            tool="RoomAgent--BookRoom", params={"room": "C.12"},
            sources=[ParamSource(param="room", value="C.12", source="tool_result",
                                 kind="exact", origin="RoomAgent--GetFreeRooms (rooms[0].id)")],
            prior_calls=1,
        )

        payload = notification.model_dump()
        assert payload["sources"][0]["origin"] == "RoomAgent--GetFreeRooms (rooms[0].id)"
        assert payload["prior_calls"] == 1


class TestFailureIsNotFatal:
    """A broken explanation must degrade to no explanation, never to a blocked call."""

    @pytest.mark.anyio
    async def test_a_malformed_response_does_not_stop_the_prompt(self):
        from src.tool_calling import ToolCaller

        class Exploding:
            @property
            def agent_messages(self):
                raise RuntimeError("boom")

        caller = ToolCaller(session=None, response=Exploding())
        notification = ConfirmActionNotification(tool="A--b", params={"x": "y"})

        caller._add_context(notification, _tool(args={"x": "y"}))

        assert notification.sources == []

    def test_no_response_means_no_context_and_no_error(self):
        from src.tool_calling import ToolCaller

        caller = ToolCaller(session=None)
        notification = ConfirmActionNotification(tool="A--b", params={"x": "y"})

        caller._add_context(notification, _tool(args={"x": "y"}))

        assert notification.sources == []
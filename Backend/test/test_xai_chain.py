"""The call chain a user can look at.

The graph is drawn from the recorded trace, so the thing to protect is that it
cannot show something that did not happen: no arrow without a resolved value
behind it, no node without a call behind it.
"""

import pytest
from fastapi.testclient import TestClient

from src.models import AgentMessage, QueryResponse, StepType, ToolCall, ToolType
from src.server import app, handle_session_id
from src.xai.chain import build_chain
from util import handle_user_session_id

app.dependency_overrides[handle_session_id] = handle_user_session_id
client = TestClient(app)


def _call(call_id, name, args=None, result=None, **kw) -> ToolCall:
    return ToolCall(id=call_id, type=ToolType.OPACA, name=name,
                    args=args or {}, result=result, **kw)


def _booking() -> QueryResponse:
    """A lookup followed by a booking that reuses one of its values."""
    return QueryResponse(query="Book a free room", agent_messages=[
        AgentMessage(agent="Tool Generator", step_type=StepType.TOOL_CALL, tools=[
            _call("m1/0", "RoomAgent--GetFreeRooms", {},
                  {"rooms": [{"id": "C.12"}]}, success=True),
        ]),
        AgentMessage(agent="Tool Evaluator", step_type=StepType.EVALUATE),
        AgentMessage(agent="Tool Generator", step_type=StepType.TOOL_CALL, iteration=1, tools=[
            _call("m3/0", "RoomAgent--BookRoom",
                  {"room": "C.12", "organizer": "Dr. Weber"}, "booked", success=True),
        ]),
        AgentMessage(agent="Output Generator", step_type=StepType.OUTPUT),
    ], content="Booked C.12.", execution_time=9.0)


class TestNodes:
    def test_one_node_per_tool_call(self):
        chain = build_chain(_booking())

        assert [n.action for n in chain.nodes] == ["GetFreeRooms", "BookRoom"]

    def test_steps_without_calls_are_counted_but_not_drawn(self):
        """The evaluator and the output generator are part of the answer,
        but a graph about where data came from has no place for them."""
        chain = build_chain(_booking())

        assert len(chain.nodes) == 2
        assert chain.steps == 4

    def test_nodes_carry_their_outcome(self):
        response = _booking()
        response.agent_messages[2].tools[0].success = False
        response.agent_messages[2].tools[0].error = "room already booked"

        chain = build_chain(response)

        assert chain.nodes[1].success is False
        assert chain.nodes[1].error == "room already booked"
        assert chain.failures == 1

    def test_nodes_carry_the_iteration_they_belong_to(self):
        chain = build_chain(_booking())

        assert chain.nodes[0].iteration == 0
        assert chain.nodes[1].iteration == 1


class TestLinks:
    def test_a_resolved_value_draws_an_arrow(self):
        chain = build_chain(_booking())

        assert len(chain.links) == 1
        link = chain.links[0]
        assert (link.from_id, link.to_id, link.param) == ("m1/0", "m3/0", "room")

    def test_a_value_the_model_supplied_draws_nothing(self):
        """No arrow is the visual form of "nothing flowed into this"."""
        chain = build_chain(_booking())

        assert not any(l.param == "organizer" for l in chain.links)
        assert chain.untraceable == 1

    def test_every_link_points_at_real_nodes(self):
        chain = build_chain(_booking())
        ids = {n.id for n in chain.nodes}

        assert all(l.from_id in ids and l.to_id in ids for l in chain.links)

    def test_a_chain_with_no_calls_is_empty_not_broken(self):
        chain = build_chain(QueryResponse(query="hello", agent_messages=[
            AgentMessage(agent="assistant", step_type=StepType.OUTPUT)
        ]))

        assert chain.nodes == [] and chain.links == []


class TestNodeDetail:
    def test_a_node_carries_where_each_argument_came_from(self):
        chain = build_chain(_booking())

        sources = {s.param: s for s in chain.nodes[1].sources}
        assert sources["room"].source == "tool_result"
        assert sources["organizer"].source == "unmatched"

    def test_the_origin_is_readable(self):
        chain = build_chain(_booking())

        origin = next(s for s in chain.nodes[1].sources if s.param == "room").origin
        assert "GetFreeRooms" in origin


class TestEndpoint:
    def test_a_missing_response_is_not_found(self):
        client.post("/chats/chain-test/query/simple", json={"user_query": ""})
        res = client.get("/chats/chain-test/responses/does-not-exist/chain")

        assert res.status_code == 404

    def test_a_missing_chat_is_not_found(self):
        res = client.get("/chats/no-such-chat/responses/x/chain")

        assert res.status_code == 404
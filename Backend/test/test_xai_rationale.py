"""Capturing why a call was made, from the model that made it.

The reason is taken in the same completion as the call, before any outcome
exists to rationalise. Two things have to hold: the field must not reach an
agent container, and asking for it must not break the call when a model
declines to answer.
"""

import pytest

from src.models import AgentMessage, QueryResponse, ToolCall, ToolType
from src.xai.chain import build_chain
from src.xai.rationale import (
    ALTERNATIVE_FIELD, RATIONALE_FIELD, add_rationale_field, extract_rationale,
    rationale_enabled,
)


def _tool_schema(**properties) -> dict:
    return {
        "name": "RoomAgent--BookRoom",
        "description": "Book a room",
        "parameters": {"type": "object", "properties": properties, "required": ["room_id"]},
    }


class TestSchemaInjection:
    def test_the_fields_are_added(self):
        patched = add_rationale_field(_tool_schema(room_id={"type": "integer"}))

        properties = patched["parameters"]["properties"]
        assert RATIONALE_FIELD in properties
        assert ALTERNATIVE_FIELD in properties

    def test_the_real_parameters_survive(self):
        patched = add_rationale_field(_tool_schema(room_id={"type": "integer"}))

        assert patched["parameters"]["properties"]["room_id"] == {"type": "integer"}

    def test_the_fields_are_not_required(self):
        """A required field would fail the call whenever a model omits it,
        trading a working tool call for an explanation."""
        patched = add_rationale_field(_tool_schema(room_id={"type": "integer"}))

        assert patched["parameters"]["required"] == ["room_id"]

    def test_the_original_schema_is_untouched(self):
        """get_tools is called repeatedly, including inside validation."""
        original = _tool_schema(room_id={"type": "integer"})

        add_rationale_field(original)

        assert RATIONALE_FIELD not in original["parameters"]["properties"]

    @pytest.mark.parametrize("tool", [
        {"name": "A--b"},
        {"name": "A--b", "parameters": None},
        {"name": "A--b", "parameters": {"type": "object"}},
    ])
    def test_a_schema_without_properties_passes_through(self, tool):
        """Internal and MCP tools do not all look the same."""
        assert add_rationale_field(tool) == tool


class TestExtraction:
    def test_the_fields_are_removed_from_the_arguments(self):
        """An OPACA action given a parameter it does not define rejects the call."""
        args, _, _ = extract_rationale({
            "room_id": 1, RATIONALE_FIELD: "the user named this room",
        })

        assert args == {"room_id": 1}

    def test_the_reason_is_returned(self):
        _, why, considered = extract_rationale({
            "room_id": 1,
            RATIONALE_FIELD: "the user named this room",
            ALTERNATIVE_FIELD: "GetRooms, but the id was already known",
        })

        assert why == "the user named this room"
        assert "GetRooms" in considered

    def test_a_call_without_a_reason_is_unchanged(self):
        args, why, considered = extract_rationale({"room_id": 1})

        assert args == {"room_id": 1}
        assert why is None and considered is None

    def test_an_empty_reason_counts_as_none(self):
        """Models fill the field with an empty string rather than omitting it."""
        _, why, _ = extract_rationale({"room_id": 1, RATIONALE_FIELD: ""})

        assert why is None

    def test_non_dict_arguments_do_not_raise(self):
        assert extract_rationale(None) == (None, None, None)


class TestSwitch:
    def test_it_is_on_by_default(self, monkeypatch):
        monkeypatch.delenv("XAI_CAPTURE_RATIONALE", raising=False)

        assert rationale_enabled()

    @pytest.mark.parametrize("value", ["false", "0", "no", "FALSE"])
    def test_it_can_be_turned_off_for_the_benchmark(self, monkeypatch, value):
        """Adding a property to every schema can move tool-selection accuracy,
        so the claim that it does not has to be measurable."""
        monkeypatch.setenv("XAI_CAPTURE_RATIONALE", value)

        assert not rationale_enabled()


class TestItReachesTheUi:
    def test_the_chain_carries_the_reason(self):
        response = QueryResponse(query="CO2 in the Experience Hub?", agent_messages=[
            AgentMessage(agent="Tool Generator", tools=[ToolCall(
                id="m1/0", type=ToolType.OPACA, name="RoomAgent--GetRoomId",
                args={"room_name": "Experience Hub"}, result=1, success=True,
                rationale="need the numeric id before I can read a sensor",
                considered="GetRooms, but only one room was named",
            )]),
        ])

        node = build_chain(response).nodes[0]

        assert node.rationale.startswith("need the numeric id")
        assert "GetRooms" in node.considered

    def test_the_reason_is_not_fed_back_to_the_model(self):
        """without_id() builds the tool history the next prompt sees. The
        model's own words coming back as input would be new input."""
        call = ToolCall(id="m1/0", type=ToolType.OPACA, name="A--b",
                        args={"x": 1}, result="ok", rationale="because")

        assert "rationale" not in call.without_id()


class TestItNeverLooksLikeAParameter:
    """The fields are ours, and the user should not meet them.

    The debug panel and the approval prompt both read a call before it is
    invoked, so stripping at invocation is too late: the first real run showed
    `GetScheduledTasks( _why: "...", _considered: "")` in the debug view, as
    though the model had chosen to pass them.
    """

    def test_a_stripped_call_carries_only_real_parameters(self):
        args, why, _ = extract_rationale({
            "room_id": 2, RATIONALE_FIELD: "need the sensor reading",
        })
        call = ToolCall(id="m1/0", type=ToolType.OPACA, name="SensorAgent--GetCo2Level",
                        args=args, rationale=why)

        assert call.args == {"room_id": 2}
        assert call.rationale == "need the sensor reading"

    def test_call_llm_strips_before_the_call_object_exists(self):
        """Asserted against the source: the stripping has to happen where the
        ToolCall is built, since everything that displays one reads it there."""
        import pathlib

        source = (pathlib.Path(__file__).resolve().parent.parent
                  / "src" / "abstract_method.py").read_text()
        build_index = source.index("agent_message.tools.append(tool)")
        strip_index = source.index("extract_rationale(args)")

        assert strip_index < build_index

    def test_invoke_tool_still_strips_as_well(self):
        """simple parses tool calls out of text itself and never reaches
        call_llm, so the second pass is not redundant."""
        import pathlib

        source = (pathlib.Path(__file__).resolve().parent.parent
                  / "src" / "tool_calling.py").read_text()

        assert "extract_rationale(tool.args)" in source


class TestItReachesTheDebugView:
    """Robert asked for the reason to show per step, starting with the debug
    output. It rides on the tool-call message so it appears as the call is
    generated, not only once the response has finished.
    """

    def test_the_tool_call_message_carries_it(self):
        from src.models import ToolCallMessage

        message = ToolCallMessage(agent="Tool Generator", id="m1/0",
                                  name="RoomAgent--GetRoomId", args={"room_name": "x"},
                                  chat_id="c", rationale="need the id first")

        assert message.model_dump()["rationale"] == "need the id first"

    def test_it_is_optional(self):
        """Models skip the field, and older stored calls never had one."""
        from src.models import ToolCallMessage

        message = ToolCallMessage(agent="a", id="m1/0", name="A--b", chat_id="c")

        assert message.rationale is None

    def test_both_send_sites_pass_it(self):
        """call_llm covers three methods; simple parses tool calls itself."""
        import pathlib

        src = pathlib.Path(__file__).resolve().parent.parent / "src"
        for name in ["abstract_method.py", "simple/simple_routes.py"]:
            assert "rationale=tool.rationale" in (src / name).read_text(), name
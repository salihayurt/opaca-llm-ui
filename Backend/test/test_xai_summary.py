"""The narrative layer, and the guard around it.

This is the only part of the explanation a model writes, so it is the only part
that can be wrong in the way the whole design is built to avoid: fluent, and
about steps that never happened. The reference check is what stands between the
two, and most of these tests are about that check rather than about prose.
"""

import pytest

from src.models import AgentMessage, QueryResponse, StepType, ToolCall, ToolType
from src.xai.confidence import assess
from src.xai.provenance import build_edges
from src.xai.summary import (
    SummaryStep, _ModelSummary, fallback, render_trace, validate,
)
from src.xai.trace import build_trace


def _response() -> QueryResponse:
    return QueryResponse(query="CO2 in the Conference Room?", agent_messages=[
        AgentMessage(id="s1", agent="Tool Generator", step_type=StepType.TOOL_CALL, tools=[
            ToolCall(id="s1/0", type=ToolType.OPACA, name="RoomAgent--GetRoomId",
                     args={"room_name": "Conference Room"}, result=2, success=True,
                     rationale="need the id before reading a sensor"),
        ]),
        AgentMessage(id="s2", agent="Tool Evaluator", step_type=StepType.EVALUATE,
                     formatted_output={"decision": "FINISHED", "reason": "got it"}),
    ], content="1000 ppm")


def _parts(response=None):
    response = response or _response()
    trace = build_trace(response)
    edges = build_edges(trace)
    return trace, edges, assess(trace, edges)


class TestWhatTheModelIsShown:
    def test_the_log_carries_the_ids_it_must_cite(self):
        trace, edges, _ = _parts()

        rendered = render_trace(trace, edges)

        assert "[s1]" in rendered and "[s1/0]" in rendered

    def test_it_carries_where_values_came_from(self):
        trace, edges, _ = _parts()

        rendered = render_trace(trace, edges)

        assert "came from the user's question" in rendered

    def test_an_invented_value_is_stated_in_the_log(self):
        """The model cannot warn about what it was not told."""
        response = _response()
        response.agent_messages[0].tools[0].args = {"organizer": "Dr. Weber"}
        trace, edges, _ = _parts(response)

        assert "CAME FROM NOWHERE" in render_trace(trace, edges)

    def test_a_stated_reason_is_marked_as_stated(self):
        trace, edges, _ = _parts()

        assert "its stated reason:" in render_trace(trace, edges)

    def test_an_evaluator_decision_is_carried_as_a_decision(self):
        trace, edges, _ = _parts()

        assert "decided: FINISHED" in render_trace(trace, edges)


class TestReferenceChecking:
    def test_a_grounded_summary_survives(self):
        trace, _, report = _parts()
        written = _ModelSummary(headline="Looked up the room, then read its sensor.", confidence="high", steps=[SummaryStep(title="Found the room", detail="…", refs=["s1", "s1/0"])],
        )

        summary = validate(written, trace, report)

        assert summary.generated
        assert summary.steps[0].refs == ["s1", "s1/0"]

    def test_an_invented_reference_is_dropped(self):
        trace, _, report = _parts()
        written = _ModelSummary(headline="…", confidence="high", steps=[SummaryStep(title="A step", detail="…", refs=["s1", "s9/9", "s1/0"])],
        )

        summary = validate(written, trace, report)

        assert summary.steps[0].refs == ["s1", "s1/0"]
        assert summary.dropped_refs == ["s9/9"]

    def test_a_summary_of_steps_that_never_happened_is_discarded(self):
        """Past a point it is not describing the log it was given, and a fluent
        account of steps that did not happen is worse than a dull correct one."""
        trace, _, report = _parts()
        written = _ModelSummary(headline="Booked a room and sent three emails.", confidence="high", steps=[SummaryStep(title="Invented", detail="…", refs=["x1", "x2", "x3"])],
        )

        summary = validate(written, trace, report)

        assert not summary.generated

    def test_a_summary_with_no_references_is_not_discarded(self):
        """Nothing was invented; there is just nothing to check."""
        trace, _, report = _parts()
        written = _ModelSummary(headline="…", confidence="high", steps=[
            SummaryStep(title="A step", detail="…", refs=[])])

        assert validate(written, trace, report).generated


class TestConfidenceStaysOurs:
    def test_the_model_may_lower_it(self):
        trace, _, report = _parts()
        written = _ModelSummary(headline="…", steps=[], confidence="low")

        assert validate(written, trace, report).confidence == "low"

    def test_the_model_cannot_raise_it(self):
        """Letting it grade itself upward would undo the reason for computing."""
        response = _response()
        response.agent_messages[0].tools[0].success = False
        trace, _, report = _parts(response)
        assert report.level == "medium"

        written = _ModelSummary(headline="…", steps=[], confidence="high")

        assert validate(written, trace, report).confidence == "medium"


class TestFallback:
    def test_it_describes_the_chain_without_a_model(self):
        trace, _, report = _parts()

        summary = fallback(trace, report)

        assert "GetRoomId" in summary.headline
        assert not summary.generated

    def test_it_reports_failures(self):
        response = _response()
        response.agent_messages[0].tools[0].success = False
        trace, _, report = _parts(response)

        summary = fallback(trace, report)

        assert "1 failed" in summary.headline
        assert summary.confidence == "medium"

    def test_a_chain_with_no_tools_is_described_not_blanked(self):
        trace, _, report = _parts(QueryResponse(query="hi", agent_messages=[
            AgentMessage(agent="assistant", step_type=StepType.OUTPUT)], content="hello"))

        assert fallback(trace, report).headline

    def test_it_carries_the_signals_as_caveats(self):
        response = _response()
        response.agent_messages[0].tools[0].success = False
        trace, _, report = _parts(response)

        assert any("failed" in c for c in fallback(trace, report).caveats)


class TestCaching:
    def test_the_hash_identifies_what_was_explained(self):
        from src.xai.chain import build_chain

        response = _response()
        before = build_chain(response).trace_hash
        response.agent_messages[0].tools[0].name = "RoomAgent--BookRoom"

        assert build_chain(response).trace_hash != before

    def test_the_summary_is_stored_on_the_response(self):
        """Chat.responses is already persisted, so this survives a restart."""
        response = _response()
        response.xai = {"headline": "…", "generated": True}
        response.xai_hash = "abc"

        restored = QueryResponse.model_validate_json(response.model_dump_json())

        assert restored.xai["headline"] == "…"
        assert restored.xai_hash == "abc"

    def test_an_old_response_has_no_explanation_and_still_loads(self):
        stored = {"query": "q", "agent_messages": [], "iterations": 1,
                  "execution_time": 1.0, "content": "a", "error": ""}

        assert QueryResponse.model_validate(stored).xai is None


class TestTheDiscardThreshold:
    """A share alone behaves badly at small numbers, so the count matters too."""

    def test_one_bad_reference_among_several_is_a_slip_not_a_fabrication(self):
        trace, _, report = _parts()
        written = _ModelSummary(headline="…", confidence="high", steps=[
            SummaryStep(title="A step", detail="…", refs=["s1", "s1/0", "s9/9"])])

        summary = validate(written, trace, report)

        assert summary.generated
        assert summary.dropped_refs == ["s9/9"]

    def test_a_single_reference_that_is_wrong_still_discards(self):
        """Nothing it claimed is in the trace."""
        trace, _, report = _parts()
        written = _ModelSummary(headline="…", confidence="high", steps=[
            SummaryStep(title="A step", detail="…", refs=["nope", "also-nope"])])

        assert not validate(written, trace, report).generated


class TestTheExplainerConfig:
    """Constructed here because nothing else did.

    The first real run failed on config construction, not on anything the other
    tests touch: llm_field carries a regex meant for the model *string*, and it
    was applied to the LLMConfig object around it. Pydantic only complains when
    the class is built, so a test suite that never builds it never finds out.
    """

    def test_it_can_be_constructed(self):
        from src.xai.explainer import XaiConfig

        assert XaiConfig().model.model

    def test_it_defaults_to_a_small_model(self):
        """This reads a log and reorganises it. Reasoning effort buys nothing
        and costs seconds on something a user is waiting for."""
        from src.xai.explainer import XaiConfig

        assert "mini" in XaiConfig().model.model

    def test_its_schema_renders_for_the_config_ui(self):
        from src.xai.explainer import XaiConfig

        assert "model" in XaiConfig.model_json_schema()["properties"]

    def test_the_explainer_is_not_offered_as_a_way_to_answer(self):
        """It never sees the conversation. Listing it beside the four
        strategies would offer users a fifth way to fail."""
        from src.server import METHODS

        assert "xai" not in METHODS


class TestWhatIsHandedToCallLlm:
    """Two runs in a row failed on the shape of the arguments, not the logic.

    call_llm is shared with the four methods and expects what they pass. The
    explainer is the only caller written from outside that habit, so what it
    hands over is checked here rather than discovered in a log.
    """

    @pytest.mark.anyio
    async def test_messages_are_chat_messages_not_dicts(self):
        """call_llm calls model_dump on each; a dict has no such method."""
        from unittest.mock import AsyncMock
        from src.models import AgentMessage, Chat, ChatMessage, SessionData
        from src.xai.explainer import XaiExplainer

        explainer = XaiExplainer(SessionData(), Chat(chat_id="c"), QueryResponse(query=""))
        explainer.call_llm = AsyncMock(return_value=AgentMessage(agent="x"))

        await explainer.explain(_response())

        messages = explainer.call_llm.call_args.kwargs["messages"]
        assert all(isinstance(m, ChatMessage) for m in messages)

    @pytest.mark.anyio
    async def test_the_trace_goes_in_as_the_user_turn(self):
        """It is a log to describe, not a conversation to continue."""
        from unittest.mock import AsyncMock
        from src.models import AgentMessage, Chat, SessionData
        from src.xai.explainer import XaiExplainer

        explainer = XaiExplainer(SessionData(), Chat(chat_id="c"), QueryResponse(query=""))
        explainer.call_llm = AsyncMock(return_value=AgentMessage(agent="x"))

        await explainer.explain(_response())

        message = explainer.call_llm.call_args.kwargs["messages"][0]
        assert message.role == "user"
        assert "[s1/0]" in message.content

    @pytest.mark.anyio
    async def test_no_tools_are_offered(self):
        """It describes actions; it does not take any."""
        from unittest.mock import AsyncMock
        from src.models import AgentMessage, Chat, SessionData
        from src.xai.explainer import XaiExplainer

        explainer = XaiExplainer(SessionData(), Chat(chat_id="c"), QueryResponse(query=""))
        explainer.call_llm = AsyncMock(return_value=AgentMessage(agent="x"))

        await explainer.explain(_response())

        assert explainer.call_llm.call_args.kwargs["tool_choice"] == "none"

    @pytest.mark.anyio
    async def test_an_empty_answer_degrades_to_the_counted_summary(self):
        from unittest.mock import AsyncMock
        from src.models import AgentMessage, Chat, SessionData
        from src.xai.explainer import XaiExplainer

        explainer = XaiExplainer(SessionData(), Chat(chat_id="c"), QueryResponse(query=""))
        explainer.call_llm = AsyncMock(return_value=AgentMessage(agent="x"))

        summary = await explainer.explain(_response())

        assert not summary.generated
        assert summary.headline
        assert summary.error


class TestFormattedOutputShapes:
    """call_llm returns a validated model; the database returns a dict.

    Both arrive in the same field, and assuming either one broke a real run.
    """

    @pytest.mark.parametrize("shape", ["model", "dict"])
    @pytest.mark.anyio
    async def test_either_shape_is_accepted(self, shape):
        from unittest.mock import AsyncMock
        from src.models import AgentMessage, Chat, SessionData
        from src.xai.explainer import XaiExplainer

        written = _ModelSummary(headline="Looked it up.", confidence="high",
                                steps=[SummaryStep(title="Step", detail="…", refs=["s1"])])
        message = AgentMessage(agent="x")
        message.formatted_output = written if shape == "model" else written.model_dump()

        explainer = XaiExplainer(SessionData(), Chat(chat_id="c"), QueryResponse(query=""))
        explainer.call_llm = AsyncMock(return_value=message)

        summary = await explainer.explain(_response())

        assert summary.generated
        assert summary.headline == "Looked it up."


class TestTheLogShowsWhatToolsReturned:
    """A real run warned that the assistant had supplied a value the tools had
    returned: "The assistant supplied the CO2 level of 800 ppm, and the user
    may want to verify this information."

    The model was right given what it was shown. The log said the call
    succeeded and never said what it returned, so the 800 in the final answer
    had nothing behind it, and the instruction to flag values that came from
    nowhere applied. Wrong input, correct behaviour.
    """

    def _rendered(self, result=800, success=True, error=None):
        from src.models import AgentMessage, QueryResponse, StepType, ToolCall, ToolType
        from src.xai.provenance import build_edges
        from src.xai.summary import render_trace
        from src.xai.trace import build_trace

        response = QueryResponse(query="CO2 in the conference room?", agent_messages=[
            AgentMessage(id="s1", agent="Tool Generator", step_type=StepType.TOOL_CALL,
                         tools=[ToolCall(id="s1/0", type=ToolType.OPACA,
                                         name="SensorAgent--GetCo2Level",
                                         args={"room_id": 2}, result=result,
                                         success=success, error=error)]),
        ], content="The CO2 level is 800 ppm.")
        trace = build_trace(response)
        return render_trace(trace, build_edges(trace))

    def test_a_returned_value_is_in_the_log(self):
        assert "returned: 800" in self._rendered()

    def test_a_failure_still_reads_as_a_failure(self):
        rendered = self._rendered(result=None, success=False, error="timeout")

        assert "FAILED: timeout" in rendered
        assert "returned:" not in rendered

    def test_a_structured_result_survives(self):
        rendered = self._rendered(result={"rooms": [{"id": "C.12"}]})

        assert "C.12" in rendered

    def test_an_empty_result_is_shown_as_empty_not_omitted(self):
        """"Returned nothing" is a finding; a silent line is not."""
        assert "returned: {}" in self._rendered(result={})


class TestCaveatsAreComputedNotWritten:
    """The model kept warning about facts the tools had returned.

    Twice: first because the log never said what a call returned, and again
    after that was fixed, because `caveats` was a required field it felt
    obliged to fill and the instruction to flag invented values gave it a
    phrase to reach for -- "The assistant supplied the CO2 level of 1200 ppm,
    and the user may want to verify this information", about a number
    GetCo2Level had plainly returned.

    A caveat is a finding, and findings here are computed. So the model no
    longer writes them: narrating is its job, deciding what is wrong is not.
    That removes the whole class rather than the instance.
    """

    def test_the_model_is_not_asked_for_caveats(self):
        from src.xai.summary import _ModelSummary

        assert "caveats" not in _ModelSummary.model_json_schema()["properties"]

    def test_a_clean_chain_produces_no_caveats(self):
        """The case that was failing: everything sourced, nothing to warn about."""
        trace, _, report = _parts()
        written = _ModelSummary(headline="Looked it up.", confidence="high", steps=[
            SummaryStep(title="Step", detail="…", refs=["s1"])])

        assert validate(written, trace, report).caveats == []

    def test_a_real_finding_still_appears(self):
        """Removing the model's caveats must not remove the caveats."""
        response = _response()
        response.agent_messages[0].tools[0].success = False
        response.agent_messages[0].tools[0].error = "timeout"
        trace, _, report = _parts(response)
        written = _ModelSummary(headline="…", confidence="high", steps=[])

        caveats = validate(written, trace, report).caveats

        assert any("failed" in c for c in caveats)

    def test_the_prompt_says_a_returned_value_is_sourced(self):
        """The instruction that produced the wrong warning, now bounded."""
        from src.xai.summary import SYSTEM_PROMPT

        assert "never suggest the assistant made it up" in SYSTEM_PROMPT
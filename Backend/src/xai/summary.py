"""A readable account of how an answer was produced.

The last layer, and the only one that costs an LLM call. Everything under it --
what ran, where the values came from, what failed, how confident the record
warrants being -- is already computed. This turns that into prose for someone
who does not want to read a graph.

Three things keep it honest.

It is given the trace, not the conversation. The model writing this never sees
the question as a question to answer; it sees a log to describe. There is
nothing for it to solve, so there is nothing for it to rationalise.

Every step it writes must name a call id that exists. References that do not
are dropped, and if too many are dropped the whole summary is discarded in
favour of the computed one. A fluent account of steps that did not happen is
worse than a dull list of steps that did.

It may lower the confidence level and not raise it. The level comes from the
record; letting the assistant revise its own grade upward would undo the reason
for computing it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import StepType
from .confidence import ConfidenceReport, lower_only
from .provenance import DataFlowEdge
from .trace import ExecutionTrace


class SummaryStep(BaseModel):
    """One step of the account, tied to something that actually ran."""
    title: str
    detail: str
    refs: list[str]


class XaiSummary(BaseModel):
    """What the model wrote, after validation."""
    headline: str = ""
    steps: list[SummaryStep] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    confidence: str = "high"
    generated: bool = False
    dropped_refs: list[str] = Field(default_factory=list)
    # Why no model wrote this, when none did. Shown rather than swallowed: a
    # panel that silently degrades gives no way to tell a broken explainer from
    # a chain there was nothing to say about.
    error: str | None = None


class _ModelSummary(BaseModel):
    """The schema the model is asked to fill.

    Deliberately not the same class as XaiSummary: confidence and provenance
    are ours to set, and a model that could write them would be able to
    overrule the record about itself.

    No defaults on any field. OpenAI's structured outputs run in strict mode,
    which requires every property to be required, and a schema it rejects comes
    back as no output at all -- which looks exactly like the model declining to
    answer.
    """
    headline: str
    steps: list[SummaryStep]
    confidence: str


# Beyond this share of invented references the account is not describing the
# trace it was given, and a fluent description of steps that did not happen is
# worse than a dull list of steps that did.
#
# The count matters as well as the share, because a share alone behaves badly
# at small numbers: one wrong id out of three is a slip, and discarding a
# summary that is otherwise correct costs the user more than the slip does. Two
# or more is a pattern.
_MAX_DROPPED_SHARE = 0.3
_MIN_DROPPED_COUNT = 2


def render_trace(trace: ExecutionTrace, edges: list[DataFlowEdge]) -> str:
    """The log handed to the model. Facts only, each with its id.

    Written out rather than passed as JSON because the ids have to be
    unmissable: the model is asked to cite them, and a schema it has to
    navigate makes that likelier to go wrong.
    """
    lines = [f"USER ASKED: {trace.query}", "", "WHAT RAN:"]

    by_call = {}
    for edge in edges:
        by_call.setdefault(edge.to_call_id, []).append(edge)

    for step in trace.steps:
        header = f"[{step.id}] {step.step_type.value} by {step.agent}"
        lines.append(header)

        if step.step_type is StepType.EVALUATE and isinstance(step.formatted_output, dict):
            decision = step.formatted_output.get("decision")
            reason = step.formatted_output.get("reason")
            lines.append(f"    decided: {decision} -- {reason}")
        elif step.content:
            lines.append(f"    said: {step.content[:200]}")

        for call_id in step.call_ids:
            call = trace.call_by_id(call_id)
            if not call:
                continue
            if call.success:
                # The result, not just that there was one. Without it the model
                # sees a number in the final answer with nothing behind it, and
                # the instruction to flag values that came from nowhere makes it
                # warn about facts the tools actually returned. Already digested
                # by build_trace, so the size is bounded.
                outcome = f"returned: {call.result}"
            else:
                outcome = f"FAILED: {call.error}"
            lines.append(f"    [{call.id}] called {call.name} -> {outcome}")
            if call.rationale:
                lines.append(f"        its stated reason: {call.rationale}")
            for edge in by_call.get(call.id, []):
                if edge.source == "tool_result":
                    origin = f"came from {edge.from_call_name}"
                elif edge.source == "user_query":
                    origin = "came from the user's question"
                elif edge.source == "prior_argument":
                    origin = f"same value as in {edge.from_call_name}"
                elif edge.kind == "weak":
                    continue
                else:
                    origin = "CAME FROM NOWHERE - the assistant supplied it"
                lines.append(f"        {edge.to_param_path}={edge.value}: {origin}")

    lines += ["", f"FINAL ANSWER: {trace.answer[:500]}"]
    return "\n".join(lines)


SYSTEM_PROMPT = """\
You describe how an assistant produced an answer. You are given a log of what \
it actually did. Your job is to make that log readable, not to judge the answer \
or to answer the question yourself.

Rules:
- Use only what is in the log. If the log does not say why something happened, \
say that it does not, rather than supplying a reason.
- Every step you write must cite the bracketed ids it describes, in `refs`.
- Where a value "CAME FROM NOWHERE", say so in the step that used it: the \
assistant supplied it rather than reading it from anywhere. A value the log \
shows a tool returning is sourced -- never suggest the assistant made it up.
- Where a call has a stated reason, you may quote it, but attribute it: it is \
what the assistant said, not something established.
- Write for someone who did not read the log. Short sentences, no jargon.
- Four steps at most. Group related calls rather than listing every one.
"""


def validate(model_output: _ModelSummary, trace: ExecutionTrace,
             report: ConfidenceReport) -> XaiSummary:
    """Keep what the model wrote that the trace supports.

    References are checked because they are checkable: an id either appears in
    the trace or it does not. Everything else in the summary is prose we cannot
    verify, which is why the reference check has to carry the weight.
    """
    known = {s.id for s in trace.steps} | {c.id for c in trace.calls}

    dropped: list[str] = []
    steps: list[SummaryStep] = []
    for step in model_output.steps:
        kept = [r for r in step.refs if r in known]
        dropped += [r for r in step.refs if r not in known]
        steps.append(SummaryStep(title=step.title, detail=step.detail, refs=kept))

    total_refs = sum(len(s.refs) for s in model_output.steps)
    if (total_refs
            and len(dropped) >= _MIN_DROPPED_COUNT
            and len(dropped) / total_refs > _MAX_DROPPED_SHARE):
        return XaiSummary(confidence=report.level, generated=False, dropped_refs=dropped)

    return XaiSummary(
        headline=model_output.headline,
        steps=steps,
        # Not the model's. A caveat is a finding, and findings here are
        # computed: the model kept warning that the assistant had supplied
        # values the tools had plainly returned, because a required field
        # invites filling and the instruction to flag invented values gave it a
        # phrase to reach for. Narrating is its job; deciding what is wrong is
        # not.
        caveats=[s.detail for s in report.signals],
        confidence=lower_only(report.level, model_output.confidence),
        generated=True,
        dropped_refs=dropped,
    )


def fallback(trace: ExecutionTrace, report: ConfidenceReport) -> XaiSummary:
    """The summary when no model is involved: counted, never phrased.

    Shown when generation fails, when too many references were invented, and
    whenever the explainer is switched off. Useful on its own, which is why it
    is the floor rather than an error message.
    """
    calls = len(trace.calls)
    failed = len(trace.failed_calls)

    if calls == 0:
        headline = "Answered without calling any tools."
    else:
        actions = ", ".join(dict.fromkeys(c.action for c in trace.calls))
        headline = f"{calls} action(s) across {len(trace.steps)} steps: {actions}."
        if failed:
            headline += f" {failed} failed."

    return XaiSummary(
        headline=headline,
        caveats=[s.detail for s in report.signals],
        confidence=report.level,
        generated=False,
    )
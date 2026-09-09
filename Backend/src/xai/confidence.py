"""How much the chain supports the answer, from the record alone.

The obvious way to get a confidence level is to ask the model. Self-reported
confidence is biased upward and poorly calibrated, so this is computed instead
and the model, when it eventually writes a summary, may only lower it. A level
the assistant awards itself is worth nothing at exactly the moment it matters:
when it is wrong.

What this measures and what it does not. It reads whether the chain ran
cleanly, converged, and grounded its arguments -- not whether the answer is
true. A single lookup that succeeds and traces to the user's question scores
high and can still return a stale value.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..models import StepType
from .provenance import DataFlowEdge
from .trace import ExecutionTrace

Level = Literal["high", "medium", "low"]

_ORDER: dict[Level, int] = {"low": 0, "medium": 1, "high": 2}


class Signal(BaseModel):
    """One observation that lowered confidence, in the user's terms."""
    id: str
    detail: str
    caps_at: Level


class ConfidenceReport(BaseModel):
    """The level and everything that argued for it.

    The signals are the point. A bare level asks to be trusted; the list says
    what to check, which is what lets someone disagree with it.
    """
    level: Level = "high"
    signals: list[Signal] = Field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.signals


def _last_evaluation(trace: ExecutionTrace):
    evaluations = [s for s in trace.steps if s.step_type is StepType.EVALUATE]
    return evaluations[-1] if evaluations else None


def assess(trace: ExecutionTrace, edges: list[DataFlowEdge]) -> ConfidenceReport:
    """Collect what the record says against the answer.

    Each signal names a ceiling rather than a penalty. Scores that add up
    invite tuning until they produce the wanted number; a ceiling states a
    rule -- an answer built on a failed call is not a high-confidence answer,
    whatever else went right.
    """
    signals: list[Signal] = []

    if trace.error:
        signals.append(Signal(
            id="error",
            detail="The request ended with an error.",
            caps_at="low",
        ))

    failed = trace.failed_calls
    if failed:
        names = ", ".join(sorted({c.action for c in failed}))
        signals.append(Signal(
            id="failed_calls",
            detail=f"{len(failed)} action(s) failed: {names}.",
            caps_at="medium",
        ))

    corrections = trace.steps_of(StepType.CORRECTION)
    if corrections:
        signals.append(Signal(
            id="corrections",
            detail=f"{len(corrections)} action(s) had to be regenerated after being rejected.",
            caps_at="medium",
        ))

    # An evaluator that still wanted another round, on a chain that ended
    # anyway, is the clearest evidence in the trace that the answer was cut
    # short rather than finished.
    last = _last_evaluation(trace)
    if last and isinstance(last.formatted_output, dict):
        decision = str(last.formatted_output.get("decision", "")).upper()
        if decision == "CONTINUE":
            signals.append(Signal(
                id="not_converged",
                detail="The last check asked for another round, but the chain stopped there.",
                caps_at="low",
            ))

    untraceable = [e for e in edges if e.source == "unmatched" and e.kind != "weak"]
    if untraceable:
        names = ", ".join(sorted({e.to_param_path for e in untraceable}))
        signals.append(Signal(
            id="untraceable_arguments",
            detail=f"Values the assistant supplied itself: {names}.",
            caps_at="medium",
        ))

    level: Level = "high"
    for signal in signals:
        if _ORDER[signal.caps_at] < _ORDER[level]:
            level = signal.caps_at

    return ConfidenceReport(level=level, signals=signals)


def lower_only(computed: Level, proposed: str | None) -> Level:
    """Let a model revise the level downward and no further.

    Used when a summary is generated. Raising it would let the assistant
    overrule the record about itself, which is the failure the computed level
    exists to prevent.
    """
    if proposed not in _ORDER:
        return computed
    return proposed if _ORDER[proposed] < _ORDER[computed] else computed
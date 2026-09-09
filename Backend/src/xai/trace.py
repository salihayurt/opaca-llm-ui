"""Normalise a QueryResponse into one canonical shape.

The four methods differ in step count, nesting and naming, but an explanation
should not. Everything downstream reads ExecutionTrace and never a
QueryResponse, so adding a fifth method means setting step_type in it and
nothing else.

Two other jobs happen here because this is the only place the raw trace is
read: tool results are digested to a bounded size, and secrets are dropped.
Anything not in the digest cannot reach an explanation, which makes this the
single point to audit for leaks.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from ..models import QueryResponse, StepType, ToolType

# Argument and result keys whose values never appear in a trace. Matched as a
# substring of the lowercased key, so "user_password" and "apiKey" both hit.
_SECRET_KEY_FRAGMENTS = (
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "credential", "authorization", "auth_", "private_key",
)

_REDACTED = "[redacted]"

# Per-value limits inside a digested result. A tool returning a 200 kB document
# contributes the same budget as one returning a sentence.
_MAX_STRING = 300
_MAX_LIST_ITEMS = 2
_MAX_DICT_KEYS = 12
_MAX_DEPTH = 4


def _is_secret_key(key: Any) -> bool:
    return isinstance(key, str) and any(f in key.lower() for f in _SECRET_KEY_FRAGMENTS)


def digest(value: Any, depth: int = 0) -> Any:
    """Shrink a value to something a prompt can afford, preserving its shape.

    Shape is kept because it carries the information an explanation needs: that
    a result was a list of eight rooms matters, the eight room objects do not.
    """
    if depth >= _MAX_DEPTH:
        return "[…]"

    if isinstance(value, str):
        if len(value) <= _MAX_STRING:
            return value
        return value[:_MAX_STRING] + f"… (+{len(value) - _MAX_STRING} chars)"

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for i, (key, val) in enumerate(value.items()):
            if i >= _MAX_DICT_KEYS:
                out["…"] = f"(+{len(value) - _MAX_DICT_KEYS} more keys)"
                break
            out[str(key)] = _REDACTED if _is_secret_key(key) else digest(val, depth + 1)
        return out

    if isinstance(value, (list, tuple)):
        items = [digest(v, depth + 1) for v in value[:_MAX_LIST_ITEMS]]
        if len(value) > _MAX_LIST_ITEMS:
            items.append(f"… (+{len(value) - _MAX_LIST_ITEMS} more items)")
        return items

    return value


def redact_args(args: dict[str, Any]) -> dict[str, Any]:
    """Arguments are shown to the user verbatim, so secrets go before anything else."""
    return {
        key: _REDACTED if _is_secret_key(key) else digest(val)
        for key, val in args.items()
    }


class TraceCall(BaseModel):
    """One tool invocation, flattened out of the step that produced it.

    Calls are lifted to the top level and given an explicit order because
    provenance is a question about the sequence of calls, not about the steps.
    In self-orchestrated the steps are not even in a meaningful order.
    """
    id: str
    order: int
    step_id: str
    name: str
    type: ToolType
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    success: bool | None = None
    error: str | None = None
    rationale: str | None = None
    considered: str | None = None

    @property
    def agent(self) -> str:
        """The OPACA agent or MCP server, i.e. the part before the separator."""
        return self.name.split("--", 1)[0]

    @property
    def action(self) -> str:
        return self.name.split("--", 1)[-1]


class TraceStep(BaseModel):
    """One LLM invocation, stripped to what an explanation can use."""
    id: str
    order: int
    step_type: StepType
    agent: str
    model: str
    iteration: int
    parent_id: str | None = None
    content: str = ""
    # The evaluators' decisions live here as structured data. Kept rather than
    # collapsed into `content` so a reader does not have to parse prose to find
    # out whether the chain was finished or cut short.
    formatted_output: Any = None
    call_ids: list[str] = Field(default_factory=list)
    execution_time: float = 0.0
    tokens: int = 0


class ExecutionTrace(BaseModel):
    """What happened, in one shape, for any method.

    `hash` covers the parts an explanation is derived from, so a cached
    explanation can be invalidated when its subject changes.
    """
    response_id: str
    query: str
    steps: list[TraceStep] = Field(default_factory=list)
    calls: list[TraceCall] = Field(default_factory=list)
    iterations: int = 0
    execution_time: float = 0.0
    answer: str = ""
    error: str = ""
    hash: str = ""

    @property
    def failed_calls(self) -> list[TraceCall]:
        return [c for c in self.calls if c.success is False]

    @property
    def models_used(self) -> list[str]:
        seen = {s.model for s in self.steps if s.model}
        return sorted(seen)

    def call_by_id(self, call_id: str) -> TraceCall | None:
        return next((c for c in self.calls if c.id == call_id), None)

    def steps_of(self, step_type: StepType) -> list[TraceStep]:
        return [s for s in self.steps if s.step_type is step_type]


def _trace_hash(response: QueryResponse) -> str:
    """Identify the trace by what an explanation depends on.

    Deliberately not a hash of the whole response: timings differ between an
    object in memory and the same one reloaded from MongoDB, and an explanation
    that invalidates itself on every restart is a cache that never hits.
    """
    import hashlib

    parts = [response.query]
    for message in response.agent_messages:
        parts.append(f"{message.id}|{message.step_type}|{message.model}")
        for call in message.tools:
            parts.append(f"{call.id}|{call.name}|{sorted(call.args)}|{call.success}")
    parts.append(response.content)
    return hashlib.sha256("\u0000".join(parts).encode()).hexdigest()[:16]


def build_trace(response: QueryResponse) -> ExecutionTrace:
    """Normalise any method's response into an ExecutionTrace.

    Steps keep the order they were appended in. For self-orchestrated that
    order is not the logical one -- workers are appended from inside
    asyncio.gather -- which is what `parent_id` and `iteration` are for.
    Consumers must group by those rather than reading the list top to bottom.
    """
    steps: list[TraceStep] = []
    calls: list[TraceCall] = []

    for order, message in enumerate(response.agent_messages):
        call_ids = []
        for call in message.tools:
            calls.append(TraceCall(
                id=call.id,
                order=len(calls),
                step_id=message.id,
                name=call.name,
                type=call.type,
                args=redact_args(call.args),
                result=digest(call.result),
                success=call.success,
                error=call.error,
                rationale=call.rationale,
                considered=call.considered,
            ))
            call_ids.append(call.id)

        steps.append(TraceStep(
            id=message.id,
            order=order,
            step_type=message.step_type,
            agent=message.agent,
            model=message.model,
            iteration=message.iteration,
            parent_id=message.parent_id,
            content=_summarise_content(message.content),
            formatted_output=digest(message.formatted_output),
            call_ids=call_ids,
            execution_time=message.execution_time,
            tokens=message.response_metadata.get("total_tokens", 0) or 0,
        ))

    return ExecutionTrace(
        response_id=response.response_id,
        query=response.query,
        steps=steps,
        calls=calls,
        iterations=response.iterations,
        execution_time=response.execution_time,
        answer=response.content,
        error=response.error,
        hash=_trace_hash(response),
    )


_WHITESPACE = re.compile(r"\s+")


def _summarise_content(content: Any) -> str:
    """Step content is prompt-shaped: JSON blobs, repeated whitespace, prose.

    Collapsed and truncated rather than parsed. An evaluator's decision is
    already available as structured data on the message; this field only has to
    stay readable.
    """
    if not isinstance(content, str):
        return ""
    collapsed = _WHITESPACE.sub(" ", content).strip()
    if len(collapsed) <= _MAX_STRING:
        return collapsed
    return collapsed[:_MAX_STRING] + "…"
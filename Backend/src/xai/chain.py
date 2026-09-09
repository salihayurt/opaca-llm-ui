"""Assemble the call chain a user can look at.

Nothing new is derived here: the nodes are the trace's calls and the links are
the provenance edges. The module exists so that neither the route nor the
frontend has to know how those two fit together, and so that the shape the UI
draws is decided in one place.

A link is drawn only where a value was resolved to an earlier result. Values
the model supplied itself have no arrow, which is the visual form of the same
finding the approval prompt words: nothing flowed into that parameter.
"""

from __future__ import annotations

from ..models import (ChainLink, ChainNode, ChainView, ConfidenceSignal, ParamSource,
                      QueryResponse)
from .confidence import assess
from .provenance import DataFlowEdge, build_edges
from .trace import ExecutionTrace, build_trace


def _to_param_source(edge: DataFlowEdge) -> ParamSource:
    return ParamSource(
        param=edge.to_param_path,
        value=edge.value,
        source=edge.source,
        kind=edge.kind,
        origin=(f"{edge.from_call_name} ({edge.from_path})" if edge.from_call_name else None),
    )


def build_chain(response: QueryResponse) -> ChainView:
    """Turn a response into nodes and links.

    Steps that made no tool call -- evaluators, the output generator -- are
    counted but not drawn. They are part of how the answer was produced, and
    the summary layer will describe them, but they have no place in a graph
    about where data came from.
    """
    trace: ExecutionTrace = build_trace(response)
    edges = build_edges(trace)

    by_call: dict[str, list[DataFlowEdge]] = {}
    for edge in edges:
        by_call.setdefault(edge.to_call_id, []).append(edge)

    nodes = []
    for call in trace.calls:
        step = next((s for s in trace.steps if s.id == call.step_id), None)
        nodes.append(ChainNode(
            id=call.id,
            order=call.order,
            name=call.name,
            agent=call.agent,
            action=call.action,
            type=call.type.value,
            args=call.args,
            success=call.success,
            error=call.error,
            iteration=step.iteration if step else 0,
            step_agent=step.agent if step else '',
            sources=[_to_param_source(e) for e in by_call.get(call.id, [])],
            rationale=call.rationale,
            considered=call.considered,
        ))

    links = [
        ChainLink(
            from_id=edge.from_call_id,
            to_id=edge.to_call_id,
            param=edge.to_param_path,
            kind=edge.kind,
        )
        for edge in edges
        if edge.source == "tool_result" and edge.from_call_id
    ]

    report = assess(trace, edges)

    return ChainView(
        response_id=trace.response_id,
        query=trace.query,
        nodes=nodes,
        links=links,
        steps=len(trace.steps),
        failures=len(trace.failed_calls),
        untraceable=sum(1 for e in edges if e.source == "unmatched" and e.kind != "weak"),
        models=trace.models_used,
        execution_time=trace.execution_time,
        trace_hash=trace.hash,
        confidence=report.level,
        signals=[ConfidenceSignal(id=s.id, detail=s.detail, caps_at=s.caps_at)
                 for s in report.signals],
    )
"""Explainability layer.

Reads the execution trace SAGE produces and derives what can be established
without asking a model anything. See docs/xai.md.
"""

from .trace import ExecutionTrace, TraceCall, TraceStep, build_trace
from .chain import build_chain
from .confidence import ConfidenceReport, assess, lower_only
from .explainer import XaiConfig, XaiExplainer, explain_response
from .summary import XaiSummary
from .rationale import (RATIONALE_FIELDS, add_rationale_field, extract_rationale,
                        rationale_enabled)
from .provenance import DataFlowEdge, build_edges, edges_for_call, summarise

__all__ = [
    "ExecutionTrace", "TraceCall", "TraceStep", "build_trace",
    "DataFlowEdge", "build_edges", "edges_for_call", "summarise",
    "build_chain",
    "ConfidenceReport", "assess", "lower_only",
    "XaiConfig", "XaiExplainer", "explain_response", "XaiSummary",
    "RATIONALE_FIELDS", "add_rationale_field", "extract_rationale", "rationale_enabled",
]
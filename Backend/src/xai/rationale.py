"""Capture why a call was made, from the model that made it.

Asking a model afterwards why it did something produces a fluent account that
need not be the real one: Turpin et al. (NeurIPS 2023) showed chain-of-thought
explanations systematically misstate the actual cause, with accuracy dropping
by up to 36% when a biased input pushes toward a wrong answer. So the reason is
taken at the moment of the decision instead, before any outcome exists to
rationalise.

The mechanism is a field added to every tool's parameter schema. The model
fills it while generating the call, inside the normal function-calling channel,
and it is stripped before the call reaches OPACA or an MCP server.

A text preamble would have been the obvious alternative and does not work here.
`call_llm` breaks out of the stream on the first OUTPUT_TEXT_DELTA when
tool_choice is "only", which tool-llm's Tool Generator uses: RESPONSE_COMPLETED
is then never processed, no tool calls are collected, and the method falls
through to output generation. Tool calling would stop, silently.
"""

from __future__ import annotations

import os
from typing import Any


def rationale_enabled() -> bool:
    """Whether to ask for a rationale at all.

    A switch rather than a constant because adding a property to every tool
    schema is not behaviourally free: it can move tool-selection accuracy in
    either direction, and the claim that this layer does not disturb the answer
    has to be measured against the benchmark rather than asserted.
    """
    return os.getenv("XAI_CAPTURE_RATIONALE", "true").strip().lower() not in ("0", "false", "no")

# Underscored so it cannot collide with a real OPACA or MCP parameter, and
# recognisable in a log when one leaks somewhere it should not.
RATIONALE_FIELD = "_why"
ALTERNATIVE_FIELD = "_considered"

RATIONALE_FIELDS = (RATIONALE_FIELD, ALTERNATIVE_FIELD)

_RATIONALE_SCHEMA = {
    RATIONALE_FIELD: {
        "type": "string",
        "description": (
            "One short sentence: why this action, with these parameter values, right now. "
            "State what you are trying to find out or achieve, not what the action does."
        ),
    },
    ALTERNATIVE_FIELD: {
        "type": "string",
        "description": (
            "The action you considered instead and why you did not choose it. "
            "Leave empty if there was no real alternative."
        ),
    },
}


def add_rationale_field(tool: dict) -> dict:
    """Return a copy of a tool schema carrying the rationale fields.

    Not marked required. A required field would make every call fail schema
    validation when a model omits it, which trades a working tool call for an
    explanation -- the wrong way round for something optional.
    """
    parameters = tool.get("parameters")
    if not isinstance(parameters, dict):
        return tool

    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return tool

    patched = dict(tool)
    patched["parameters"] = {
        **parameters,
        "properties": {**properties, **_RATIONALE_SCHEMA},
    }
    return patched


def extract_rationale(args: dict[str, Any]) -> tuple[dict[str, Any], str | None, str | None]:
    """Split the rationale out of a call's arguments.

    Returns the arguments as they should be invoked, plus what the model said.
    The fields never reach an agent container: they are ours, and an OPACA
    action that received an unexpected parameter would reject the call.
    """
    if not isinstance(args, dict):
        return args, None, None

    cleaned = {k: v for k, v in args.items() if k not in RATIONALE_FIELDS}
    why = args.get(RATIONALE_FIELD) or None
    considered = args.get(ALTERNATIVE_FIELD) or None
    return cleaned, why, considered
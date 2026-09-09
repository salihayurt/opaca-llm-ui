"""The one component here that calls a model.

Kept out of the METHODS registry deliberately. It is not a way to answer a
question -- it never sees the conversation, only a log -- and listing it beside
the four strategies would offer users a fifth way to fail.

Subclasses AbstractMethod anyway, for what that base class already solves:
model configuration, API keys, missing-key prompts, streaming and metrics.
Reimplementing those to avoid an inheritance that reads oddly would be the
worse trade.
"""

from __future__ import annotations

import logging
from ..abstract_method import AbstractMethod
from ..models import (Chat, ChatMessage, LLMConfig, MethodConfig, QueryResponse, SessionData,
                      StepType)
from .chain import build_chain
from .confidence import assess
from .provenance import build_edges
from .summary import SYSTEM_PROMPT, XaiSummary, _ModelSummary, fallback, render_trace, validate
from .trace import build_trace

logger = logging.getLogger(__name__)


class XaiConfig(MethodConfig):
    """Configuration for the explainer.

    A separate model from the task-solving ones, and defaulted to a small fast
    one. This reads a log and reorganises it; reasoning effort buys nothing and
    costs seconds on something a user is waiting for.
    """
    model: LLMConfig = MethodConfig.llm_role(
        title="Explanation model",
        description=("Writes the explanation. It reads the recorded trace and never the "
                     "conversation, so it needs no tools and no reasoning effort -- the job "
                     "is to reorganise a log. Needs a host that supports structured output."),
    )


class XaiExplainer(AbstractMethod):
    NAME = "xai"
    CONFIG = XaiConfig

    async def query(self, message: str) -> QueryResponse:
        """Not a way to answer anything. Present because the base class is abstract."""
        raise NotImplementedError("XaiExplainer explains a response; it does not produce one.")

    async def explain(self, target: QueryResponse) -> XaiSummary:
        """Describe how `target` was produced.

        Never raises. An explanation that fails should leave the user with the
        counted summary, which is useful on its own, rather than an error where
        an explanation was expected.
        """
        trace = build_trace(target)
        edges = build_edges(trace)
        report = assess(trace, edges)

        try:
            config: XaiConfig = self.get_config()
            message = await self.call_llm(
                model_config=config.model,
                agent="XAI Explainer",
                system_prompt=SYSTEM_PROMPT,
                # The trace goes in as the user turn: the model is being asked
                # to describe this log, not to continue the conversation it
                # records.
                messages=[ChatMessage(role="user", content=render_trace(trace, edges))],
                tool_choice="none",
                response_format=_ModelSummary,
                status_message="Explaining",
                step_type=StepType.OUTPUT,
            )
            if not message.formatted_output:
                degraded = fallback(trace, report)
                degraded.error = "The explanation model returned nothing."
                return degraded

            # call_llm hands back an already-validated model when a
            # response_format was given, but the same field arrives as a plain
            # dict once a response has been through the database. Accept both:
            # model_validate takes either.
            written = _ModelSummary.model_validate(message.formatted_output)
            return validate(written, trace, report)
        except Exception as exc:
            logger.warning(f"Could not generate an explanation: {exc}", exc_info=True)
            degraded = fallback(trace, report)
            degraded.error = f"{type(exc).__name__}: {exc}"
            return degraded


async def explain_response(session: SessionData, chat: Chat, target: QueryResponse) -> XaiSummary:
    """Return the summary for one response, generating it once.

    Cached on the response itself, keyed by the trace hash. The response is
    already persisted with the chat, so the explanation survives a restart for
    free, and a hash mismatch means the thing being explained changed.
    """
    chain = build_chain(target)
    if target.xai is not None and target.xai_hash == chain.trace_hash:
        return XaiSummary(**target.xai)

    # A QueryResponse of its own, so the explainer's own tokens and timings are
    # recorded somewhere rather than charged to the answer it describes.
    scratch = QueryResponse(query="")
    explainer = XaiExplainer(session, chat, scratch, streaming=False)
    summary = await explainer.explain(target)

    target.xai = summary.model_dump()
    target.xai_hash = chain.trace_hash
    return summary
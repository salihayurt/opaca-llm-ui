# XAI in SAGE — Design

Status: design accepted. Step 1 (trace enrichment) is implemented; the
remaining steps are listed in Section 7 and are not built yet. Latency and cost
figures will be filled in from measurement, not estimated.

---

## 1. What is being explained

Not the model's weights — the system's **actions**. Concretely:

> Why did SAGE invoke action *A* with parameters *P* at step *N*, and did the
> resulting chain actually answer the question?

This is the only branch available to us: SAGE calls GPT, Claude, Gemini and
Mistral over the network through LiteLLM, so there are no logits, activations or
gradients to inspect. Attention maps and SHAP values are out of reach, and would
not help a user decide whether to let an agent book a room even if they were not.

It is also the branch the literature calls *explainable agency*: an autonomous
system reporting what goals it pursued, what it did, and what happened
(Langley et al. 2017). Zhao et al. (ACM TIST 15(2), 2024) place it in the
behavioural rather than mechanistic half of LLM explainability.

## 2. The commitment that shapes everything else

**We never ask the model, after the fact, why it did something.**

Turpin et al. (NeurIPS 2023) showed that post-hoc chain-of-thought explanations
systematically misrepresent the real cause of a model's output: bias the input
toward a wrong answer and models produce fluent rationalisations that never
mention the bias, with accuracy dropping by up to 36% on BIG-Bench Hard.
Plausible is not faithful.

So the explanation is assembled from **what was recorded**, and where a reason is
needed, it is captured **at the moment the decision is made** rather than
reconstructed afterwards.

## 3. Three tiers, labelled differently in the UI

A statement's trustworthiness depends on where it came from, and mixing tiers in
one paragraph of prose is how a reconstruction gets read as a fact.

| Tier | What | Source | Guarantee |
|---|---|---|---|
| **T1 — fact** | Which tools ran, with what arguments, what came back, what failed, how long it took, which model ran which step | Read from the trace. No LLM. | True by construction |
| **T1.5 — provenance** | Where each parameter value came from: the user's query, an earlier tool result, a retrieved document — or nowhere | Deterministic value matching. No LLM. | True by construction |
| **T2 — stated reason** | Why the generating model chose this action, and which alternative it rejected | Emitted by the generating model *in the same completion as the tool call* | Self-report, but not post-hoc: the outcome does not exist yet, so there is nothing to rationalise |
| **T3 — reconstruction** | A narrative over the whole chain | A separate explainer model reading the trace | Plausible; labelled as interpretation |

T1.5 is the part worth dwelling on. Classical local attribution (LIME, SHAP,
Integrated Gradients) exists to *approximate* which input caused an output for a
model whose internals are hidden. In a tool-calling agent the same question —
where did `2026-08-27` in this booking come from? — is not an approximation
problem: the value either appears in an earlier result, or in the user's message,
or nowhere. It is computed exactly, in Python.

The `nowhere` case is the payoff. A parameter that traces to no source is, by
construction, one the model invented, and that is flagged rather than narrated.

## 4. Where it appears

Nothing about the answer path changes. The explanation is opt-in everywhere.

- **Approval prompt.** When a tool's approval state is `ask`, the confirmation
  dialog shows the parameter sources, the generating model's stated reason, and
  what has happened so far. All of that is free — no LLM call. This is the most
  valuable placement: it is the only moment where a user can still intervene, the
  action has not run so there is no outcome to rationalise, and the user is
  already waiting.
- **Call chain.** A graph of the tool calls, with data-flow edges drawn from the
  provenance. Clicking a node, or a single parameter, explains that one thing.
- **Chain assessment.** Whether the chain actually satisfied the request, with a
  confidence level that deterministic signals may only lower.
- **Summary panel.** One explainer call, on demand, from a footer icon next to
  the existing copy/debug/tools/metrics icons.

Everything deterministic is computed eagerly because it costs nothing. Every LLM
call is gated behind a click.

## 5. Why optional rather than always shown

Bansal et al. (CHI 2021) found explanations increased acceptance of AI
suggestions *regardless of whether the suggestion was correct*. Vasconcelos et
al. (2023) found they reduce overreliance only when they let the user verify
cheaply. An always-on explanation manufactures trust; an available one, anchored
in checkable references, supports it.

Miller (2019) is the reason for progressive disclosure rather than a trace dump:
human explanations are selective and contrastive, not exhaustive causal listings.

## 6. What Step 1 added

The trace SAGE already produced was nearly sufficient. Two things were missing,
and both are now recorded rather than inferred.

**A step's role.** `AgentMessage.agent` is a display string that differs per
method (`Tool Generator`, `assistant`, `WorkerAgent`, `IterationAdvisor`), so
anything reading the trace would have had to parse four naming schemes.
`StepType` records the role directly — `routing`, `plan`, `tool_call`,
`correction`, `evaluate`, `output`, `error` — so the explanation is built from
what a step *did*.

Alongside it, `model`, `iteration` and `parent_id`. The last matters most in
`self-orchestrated`, where worker steps are appended from inside
`asyncio.gather`: list order there does not reflect the logical structure, and a
reader without a parent link would confidently narrate a sequence that never
happened.

**How a tool call ended.** Errors are written into `ToolCall.result` as prose by
four separate paths in `invoke_tool`, so nothing downstream could tell a failure
from a result that merely mentions one — a tool reporting on errors returns text
full of the word. `success` and `error` are now set at the single point every
path returns through, matching the failure prefix rather than searching for it
anywhere in the result.

Two constraints held throughout:

- The new fields stay out of `ToolCall.without_id()`, which feeds tool history
  back to the LLM. Adding fields to what the model sees changes how the methods
  behave, and an explainability layer that changes task-solving behaviour is not
  an explainability layer.
- Every field has a default, so sessions stored in MongoDB before these fields
  existed still deserialize.

## 7. Remaining steps

Ordered so that the free, verifiable parts land before anything that costs a
token.

| # | Step | LLM cost |
|---|---|---|
| 1 | Trace enrichment | none | ✅ done |
| 2 | Trace normaliser + parameter provenance | none |
| 3 | Rationale capture (`_why` / `_considered`) in the tool schema, with a benchmark A/B | ~35 tok/call |
| 4 | Citation verification for RAG answers | none |
| 5 | Enriched approval dialog | none |
| 6 | Deterministic confidence signals | none |
| 7 | Call-chain visualisation | none |
| 8 | On-demand summary, per-node explanations, follow-up questions | one call each, on click |

Steps 1–7 give a working explanation layer that cannot hallucinate, because no
model is involved in producing it. Step 8 adds narrative on top.

### On rationale capture (step 3)

`AbstractMethod.call_llm` aborts the stream on the first text delta when
`tool_choice="only"`, which `tool-llm`'s Tool Generator uses. A ReAct-style
reasoning preamble would therefore silently disable tool calling — the stream
ends before `RESPONSE_COMPLETED`, no tool calls are collected, and the method
falls through to output generation. The rationale is instead carried as an extra
property in the tool schema, stripped before invocation, and `check_valid_action`
has to whitelist it or the correction loop will fire on every call.

Adding a field to every tool schema is not behaviourally free, which is why this
step ships with a benchmark comparison rather than an assurance.

### On logprobs

Considered and rejected for the product. Tool selection is a discrete choice, so
the distribution over the tool-name position would be a genuinely contrastive
answer to "why not the other one" — but the API does not expose it. Responses API
logprobs are scoped to assistant output text
(`include: ["message.output_text.logprobs"]`), and a function call is not output
text; Chat Completions has long returned `content=None` for logprobs when
function calling is enabled. Anthropic exposes none at all, so the feature would
vanish on most configured hosts.

Where it is feasible is self-hosted vLLM, which exposes logprobs fully. That
makes a measurement rather than a feature: comparing the model's self-reported
confidence against its actual decision distribution tells us whether the stated
confidence is calibrated. Worth doing as an experiment, not as a dependency.

## 8. How this is evaluated

1. **Dangling-reference rate** — how often the explainer cites a step or tool
   call that does not exist in the trace. The faithfulness proxy, per method and
   per explainer model.
2. **Provenance coverage** — what fraction of tool parameters resolve to a source
   versus `unmatched`, per method. A method with many unmatched parameters is a
   method that invents arguments; this is a useful diagnostic for SAGE
   independently of XAI.
3. **T2 vs T3 agreement** — does the post-hoc reconstruction match the reason the
   generating model actually stated? Systematic divergence is the Turpin result
   reproduced inside SAGE on SAGE's own data.
4. **Cost and latency overhead**, with `self-orchestrated` as the stress case.
5. **Benchmark A/B on the schema change** from step 3, against
   `correct_tool_usage`.

# XAI in SAGE

Status: built. Evaluation numbers are not in yet; §10 says which are planned.

---

## 1. Scope

Explanandum: the decision to invoke action *A* with parameters *P* at step *N*,
and the chain of such decisions.

| Approach | Applicable | Reason |
|---|---|---|
| Attention maps, gradient attribution, SHAP/LIME over tokens, probing | No | Models are served over HTTP via LiteLLM. No logits, activations or gradients. |
| Natural-language explanation, provenance, plan/trace summarisation | Yes | Operates on data SAGE already produces. |

Classification: behavioural branch of LLM explainability (Zhao et al. 2024);
explainable agency (Langley et al. 2017).

## 2. The decision the rest follows from

**The model is never asked afterwards why it did something.**

Turpin et al. (NeurIPS 2023) showed post-hoc chain-of-thought explanations
systematically misstate the actual cause: bias the input toward a wrong answer
and models produce fluent rationalisations that never mention the bias, with
accuracy dropping up to 36% on BIG-Bench Hard.

So: what happened is read from the trace, where each value came from is
computed, and the reason for an action is taken from the generating model in the
same completion as the call — before an outcome exists to rationalise.

## 3. Evidence tiers

| Tier | Content | Producer | Guarantee |
|---|---|---|---|
| T1 | Tools invoked, arguments, results, failures, retries, timings, model per step | trace read, no LLM | exact |
| T1.5 | Source of each argument: user query, prior result, prior argument, or nothing | deterministic matching, no LLM | exact |
| T2 | Why this action, which alternative was rejected | generating model, with the call | self-report, pre-outcome |
| T3 | Narrative over the chain | separate explainer model | reconstruction, reference-checked |

The UI renders them differently: T1.5 stated plainly, T2 quoted, T3 labelled as
written by a model. Eiband et al. (CHI EA 2019) found explanations with no
information content still raise trust, so a reconstruction that looks like a
fact is a trust placebo.

### Why T1.5 is the interesting one

LIME and SHAP approximate which input caused an output because the causal path
runs through weights nobody can inspect. In a tool-calling agent it runs through
data: `room_id=1` occurs in an earlier result, or in the user's message, or
nowhere. Decidable, so decided rather than estimated — and "nowhere" identifies
an argument the model supplied itself, which no approximation method isolates.

## 4. Where it appears

| Surface | Trigger | Tiers | LLM cost |
|---|---|---|---|
| Confirmation prompt | `ToolApprovalState.ASK` | T1.5, T2 | 0 |
| Call chain graph | footer icon | T1, T1.5 | 0 |
| Node detail | click a node | T1.5, T2 | 0 |
| Confidence and its signals | shown with the chain | T1 | 0 |
| Written summary | "Explain in words" | T3 | 1 call, cached |

The confirmation prompt is the placement that matters: the only moment a user
can still intervene, the action has not run so there is no outcome to
rationalise, and they are already waiting, so the explanation costs nothing.

## 5. Design decisions

| # | Decision | Reason |
|---|---|---|
| D1 | Derive from the recorded trace; never ask the model afterwards | Turpin et al. 2023 |
| D2 | Rationale captured in the tool schema at generation time | Same-completion emission precedes the outcome; ~35 tokens against a full extra call |
| D3 | Provenance computed deterministically | Exact, free, cannot hallucinate |
| D4 | Tiers rendered differently | Eiband et al. 2019 |
| D5 | Opt-in, never rendered by default | Bansal et al. 2021: explanations raise acceptance regardless of correctness |
| D6 | Confidence computed; a model may lower it, never raise it | Self-reported confidence is biased upward |
| D7 | No numeric confidence | False precision |
| D8 | Summary generated on request, cached on the response | Most answers are never questioned |
| D9 | Stored as `QueryResponse.xai` | `Chat.responses` already persists and is already returned; no new store, correct deletion, and it does not reach the model since `Chat.messages` projects only query and content |
| D10 | New fields excluded from `ToolCall.without_id()` | That feeds tool history to the LLM; adding fields changes prompts, i.e. changes behaviour |
| D11 | Counterfactuals out of scope | Requires re-execution; OPACA actions have side effects |
| D12 | No 4th agent in `tool-llm` | A judge model gives a second opinion, not the reason, and costs a call per iteration |
| D13 | Logprobs not used | §9 |
| D14 | The stated reason is shown to the user and withheld from the models | §6.4 |

## 6. What is built

### 6.1 Trace (`src/models.py`, `abstract_method.py`, `tool_calling.py`)

`StepType` — `ROUTING`, `PLAN`, `TOOL_CALL`, `CORRECTION`, `EVALUATE`,
`OUTPUT`, `ERROR`. `agent` is a per-method display string, so the role is
recorded rather than parsed out of four naming schemes.

| Field | Purpose |
|---|---|
| `AgentMessage.step_type` | semantic role |
| `AgentMessage.model` | model per step; previously only token usage survived |
| `AgentMessage.iteration` | round index |
| `AgentMessage.parent_id` | spawning step; required in `self-orchestrated`, where workers are appended from inside `asyncio.gather` and list order says nothing |
| `ToolCall.success` / `error` | `None` means not invoked, which is distinct from failed |
| `ToolCall.rationale` / `considered` | the generating model's stated reason |
| `QueryResponse.response_id` | stable address; an index shifts |
| `QueryResponse.xai` / `xai_hash` | the cached summary and what it explains |

All optional with defaults, so sessions stored before them still load.

Failures are classified by prefix, not substring: every failure path in
`invoke_tool` writes `"Failed to invoke … tool."` at the start, while a tool
reporting on errors returns the same words mid-text.

The ~17 `call_llm` sites across the four methods pass their step type.
`simple` and `simple-tools` use one call for both jobs, so theirs is set after
the fact from whether tools came back.

### 6.2 Normaliser (`src/xai/trace.py`)

`build_trace(response) -> ExecutionTrace`. Steps in append order; tool calls
lifted to the top level with an explicit sequence, because provenance is a
question about the order of calls, not the shape of the step tree.

Also the only place the raw trace is read, so it is where results are digested
to a bounded size and secret-looking keys are dropped. Nothing outside the
digest can reach an explanation.

`hash` covers query, step ids, types, models, call names, argument keys and
outcomes — not timings, which differ between a live object and the same one
reloaded, and would invalidate every cached summary on restart.

### 6.3 Provenance (`src/xai/provenance.py`)

Each argument resolves to `user_query`, `tool_result`, `prior_argument` or
`unmatched`, with a match kind of `exact`, `normalised`, `substring`,
`reworded` or `weak`. Only earlier calls are searched.

Four rules that real traces forced:

- **Diacritics are folded.** Users type ASCII, models answer accented: a search
  for `nasil calisan` came back as `nasıl çalışan` and looked invented. German
  umlauts and ß matter for the deployment.
- **Reworded values count as the user's.** *"What is the part number for the air
  filter?"* becomes `air filter part number`. Every word must occur in the
  query, so a new term cannot be smuggled into an otherwise matching one.
- **A single-value result links even for a small number.** `GetRoomId` returns
  `1` and that `1` becomes `room_id` — the most common chain the smart-office
  agents produce. Small integers are otherwise skipped as uninformative, which
  is right for a `1` found inside a large result and wrong for one that *is* the
  result.
- **`prior_argument` is not grounding.** A value reused from an earlier call's
  arguments says where it was seen, not where it came from. Without it every
  carried-forward value read as invented, and a warning that fires on most
  parameters is one nobody reads.

`edges_for_call` resolves a call that has not run, for the confirmation prompt.
Whether the pending call is already in the trace depends on the method, so both
are handled.

### 6.4 Rationale (`src/xai/rationale.py`)

`_why` and `_considered` are added to every tool schema in `get_tools`, the one
place all four methods pass through. Not required: a required field would fail
the call whenever a model omits it.

They are stripped where the `ToolCall` is built, not where it is invoked — the
debug view and the confirmation prompt both read a call before invocation, and
the first run showed `GetScheduledTasks( _why: "…" )` as though the model had
chosen to pass it. `invoke_tool` strips again, because `simple` parses tool
calls out of text itself and never goes through `call_llm`.

`check_valid_action` needed no whitelist: it re-reads `get_tools`, so the fields
are in the schema it validates against.

**A text preamble is not an alternative.** `call_llm` breaks out of the stream on
the first `OUTPUT_TEXT_DELTA` when `tool_choice="only"`, which `tool-llm`'s
generator uses. `RESPONSE_COMPLETED` is never processed, no tool calls are
collected, and the method falls through to output generation. Tool calling would
stop, silently.

`XAI_CAPTURE_RATIONALE=false` disables it. Adding a property to every tool
schema can move which tool a model picks, in either direction, so the claim that
this layer leaves the answer alone needs a switch to measure against rather than
an assurance.

**Shown to the user, withheld from the models.** `ToolCall.without_id()` builds
the tool history the next prompt reads, and the rationale is excluded from it,
so the Output Generator never sees why a call was made.

Asked whether to pass it to the Output Generator as well. Three reasons not to.
Everything else here rests on the answer being unchanged by this layer, and a
reason in the prompt ends that. The Output Generator already narrates its own
tool use unprompted — *"This information was retrieved using the following
tools"* — and giving it more material for that narration works against what is
being built: an account that is checkable and kept separate from the answer,
rather than mixed into it. And a stated reason shaping the answer it was meant
to describe is the loop the whole design exists to avoid.

Tested against `_build_tool_desc`, the prompt the Output Generator is actually
given, rather than against `without_id()` — the guarantee is about what the
model reads, and a later change to either could break it silently.

### 6.5 Confidence (`src/xai/confidence.py`)

Signals, each naming a ceiling rather than a penalty: failed calls (medium),
regenerated calls (medium), untraceable arguments (medium), a last evaluation
that still said `CONTINUE` (low), a request that ended in error (low).

Ceilings rather than points because scores that add up invite tuning until they
produce the wanted number. An answer built on a failed call is not a
high-confidence answer whatever else went right.

Only the *last* evaluation counts. A `CONTINUE` mid-loop is the loop working; one
at the end is a chain cut short.

The signals are shown, not just the level. A bare level asks to be trusted; the
list says what to check.

### 6.6 Chain view (`src/xai/chain.py`, `Frontend/src/components/XaiChain.vue`)

`GET /chats/{id}/responses/{response_id}/chain`. Nodes are tool calls; links are
provenance edges resolved to an earlier result. Steps that made no call are
counted but not drawn.

An argument the model supplied itself has nothing pointing at it — the visual
form of what the confirmation prompt states in words.

Laid out by depth in the data-flow graph, not by iteration: calls that depend on
each other can share a round, and then every node lands in one column with the
arrows folded underneath. Hand-rolled SVG; chains run to a handful of nodes and
a graph library would cost more than it saves.

### 6.7 Written summary (`src/xai/summary.py`, `explainer.py`)

`POST /chats/{id}/responses/{response_id}/explain`. One call, cached on the
response by trace hash.

The model is given the trace, not the conversation: a log to describe, not a
question to answer. Every step it writes must cite an id that exists; invented
ones are dropped, and a *pattern* of them discards the summary for the counted
one. The threshold needs a count as well as a share — one wrong id in three is a
slip, and discarding an otherwise correct summary costs more than the slip does.

It may lower the confidence level, never raise it.

`XaiExplainer` subclasses `AbstractMethod` for model config, API keys and
metrics, but is kept out of `METHODS`: it never sees the conversation, and
listing it beside the four strategies would offer a fifth way to fail.

The counted summary is the floor — shown when generation fails, when too many
references were invented, and useful on its own. When generation failed, the
panel says why: a panel that quietly degrades gives no way to tell a broken
explainer from a chain with nothing to say about it.

### 6.8 Three seams with `call_llm`

Recorded because each cost a round trip and each now has a test. `call_llm` is
shared with the four methods and expects what they pass; the explainer is the
only caller written from outside that habit.

- `messages` must be `ChatMessage`, not dicts — `model_dump` is called on each.
- A nested `LLMConfig` uses `MethodConfig.llm_role`, not `llm_field`, which
  carries a regex meant for the model *string*.
- `formatted_output` is an already-validated model when a `response_format` was
  given, and a plain dict once it has been through the database.

## 7. Tests

`Backend/test/test_xai_*.py`, around 130 tests.

`test_xai_step_types.py` reads the source for `call_llm` sites that forgot to
declare their step type. A missed site is silent — the default is valid, nothing
fails, the trace is just wrong — and it caught two while being written.

## 8. RAG interaction

`SearchDocuments` is an internal tool, so retrieval appears in the trace as a
`ToolCall` with no separate path. Two changes in `src/rag/service.py` would help
and are not done: `format_context` drops `StoredChunk.score`, a free
retrieval-quality signal; and keeping chunk text or a hash per source entry
would let a citation be verified afterwards without re-querying Qdrant, whose
index may have changed.

## 9. Logprobs — rejected

Principled at one point: tool selection is a discrete choice, so the
distribution at the tool-name position would answer "why not the alternative"
from the decision distribution rather than a self-report.

| Blocker | Detail |
|---|---|
| Not exposed for tool calls | Responses API scopes logprobs to `include: ["message.output_text.logprobs"]`; a function call is not output text. Chat Completions returns `content=None` with function calling enabled. |
| No output text to attach to | `tool-llm`'s generator runs with `tool_choice="only"`. |
| Provider coverage | Anthropic exposes none; Gemini and Mistral partial. |
| Tokenisation | Tool names are multi-token and share prefixes. |
| Calibration | RLHF models are miscalibrated; values shift with temperature and are not reproducible. Single-sample logprobs are a weak hallucination signal; semantic entropy needs N samples. |

Feasible on self-hosted vLLM as a measurement: compare `_confidence` against the
observed decision distribution and against `correct_tool_usage`.

`_considered` is the substitute in the product — a contrastive self-report for
about ten tokens, on every provider.

## 10. Evaluation

Nothing here is measured yet. The layer was built and exercised by hand; these
are the numbers that would turn its claims into evidence, and the reason each
one is the one to collect.

| Metric | Definition | Why |
|---|---|---|
| Dropped-reference rate | Share of explainer references absent from the trace | Faithfulness proxy for T3 |
| Provenance coverage per method | Traced vs `unmatched`, per strategy | A method with many unmatched arguments invents them; useful beyond XAI |
| T2/T3 agreement | Does the reconstruction match the stated reason | Turpin's result measured inside SAGE |
| Overhead | p50/p95 generation time and cost | `self-orchestrated` is the stress case |
| Schema A/B | `correct_tool_usage` with `XAI_CAPTURE_RATIONALE` on and off | Makes "does not disturb the answer" measurable rather than asserted |

The last one matters most and is the cheapest: two benchmark runs against the
same platform, differing only in the flag. Same questions in both, so the
comparison should be paired — with a few dozen questions, a difference of two is
what re-running the same condition twice produces, and reading it as an effect
would be worse than not measuring at all.

## 11. Known issues

`models.py` `add_mcp_server` uses `tool.inputSchema`; current `mcp` releases
renamed it to `input_schema`, so `test_mcp.py` fails with `AttributeError`.
Pre-existing, unrelated.

Stored chats contain responses with zero steps, paired with a full one for the
same query — `query_chat` appends the response before running it, so aborted
generations leave an empty record. The explanation layer handles them; they look
odd in history.

## 12. References

- Zhao, H. et al. (2024). *Explainability for Large Language Models: A Survey.* ACM TIST 15(2). arXiv:2309.01029
- Turpin, M. et al. (2023). *Language Models Don't Always Say What They Think.* NeurIPS 36. arXiv:2305.04388
- Lanham, T. et al. (2023). *Measuring Faithfulness in Chain-of-Thought Reasoning.* arXiv:2307.13702
- Liao, Q.V., Gruen, D., Miller, S. (2020). *Questioning the AI.* CHI '20. arXiv:2001.02478
- Miller, T. (2019). *Explanation in Artificial Intelligence.* Artificial Intelligence 267.
- Langley, P. et al. (2017). *Explainable Agency for Intelligent Autonomous Systems.* AAAI/IAAI.
- Ehsan, U. et al. (2019). *Automated Rationale Generation.* IUI '19.
- Bansal, G. et al. (2021). *Does the Whole Exceed its Parts?* CHI '21.
- Vasconcelos, H. et al. (2023). *Explanations Can Reduce Overreliance on AI Systems.* CSCW.
- Eiband, M. et al. (2019). *The Impact of Placebic Explanations on Trust in Intelligent Systems.* CHI EA '19.
- Farquhar, S. et al. (2024). *Detecting Hallucinations Using Semantic Entropy.* Nature 630.
- *HANSEL: Extracting Breadcrumbs from Web Agent Trajectories.* arXiv:2606.18671
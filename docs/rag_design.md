# RAG in SAGE — Design

Status: design accepted; the cost model in Section 7 and the ablation
results in Section 6 are pending measurement and will be replaced with real
figures. Numbers attributed to the TU Berlin ablation were measured on QASPER
and MultiHop-RAG, not on SAGE.

---

## 1. What RAG solves here

| Situation | Today | RAG helps? |
|---|---|---|
| PDF or image, OpenAI-family host | Native file upload | No — RAG is worse |
| DOCX, PPTX, XLSX, CSV, TXT, MD — any host | *silently dropped* | **Yes** |
| Document larger than the context window | nothing | **Yes** |
| Several documents, find the right passage | nothing | **Yes** |
| Any document, self-hosted model | nothing | **Yes** |

Row two is a bug. `file_utils.py:upload_files` branches on
`is_image()` / `is_pdf()` and falls through to
`logger.info("Skipping file upload (Type not supported)")`. The frontend sets
no `accept` filter, so a `.docx` uploads, appears in the Files sidebar, can be
marked active — and never reaches the model, with no error shown. SAGE's own
sample prompt *"Analyze the attached CSV…"* cannot work today.

Row five: `upload_files` raises unless the host is `openai`, `azure`,
`vertex_ai` or `bedrock`. GT-ARC runs vLLM, so documents are unusable there in
any format.

## 2. Constraints

| Constraint | Evidence | Consequence |
|---|---|---|
| Prompt is already full | 13,396 tokens for one simple question | Added tools cost real budget |
| Tool selection is SAGE's own metric | `correct_tool_usage` 149/180 | Must not regress — we measure it |
| Latency budget is small | 2.6 s baseline | No blocking LLM calls in the query path |
| Corpus size varies widely | 1–5 documents, but one manual is hundreds of chunks | Adaptive, not fixed, thresholds |
| German is first-class | `default_prompts.json` ships `GB` and `DE` | English-only components are a real gap |
| No GPU | deployment target | Embeddings via API; local models must be CPU-viable |

## 3. Decisions

| Stage | Choice | Why |
|---|---|---|
| Chunking | ~300 tokens, boundary-aware | Small chunks won everywhere; `parent1024/256` scored all-hops@1k = 0.0000. Chunker mattered more than retriever |
| Contextual chunking | On, async, flagged | 67% retrieval-failure reduction reported by Anthropic. Manual chunks are context-poor: *"hold the button for 3 seconds"* matches nothing on its own |
| Retrieval | Hybrid dense + BM25, RRF | Dense-only was weakest on both benchmarks. SAGE queries are full of ids and codes (`room 7`, `E14`) that embeddings compress away |
| Reranking | On, adaptive, flagged | Improved every QASPER config; degraded MultiHop (0.4299 → 0.3122). Single-document questions dominate, so default on — but skip below a chunk-count threshold |
| Query handling | Passthrough + conversational rewrite | HyDE hurt both benchmarks. Rewrite was untested there but SAGE is multi-turn: *"and its price?"* retrieves nothing verbatim |
| Orchestration | `stuff` only | `refine` collapsed (token_f1 ≈ 0.01), `stuff` beat `map_reduce` on correctness (83% vs 75%) and cost |
| Embedding | OpenAI via LiteLLM | No GPU available. Routed through LiteLLM (a TODO already in the code) so GT-ARC can switch to a self-hosted model with one env var — a data-sovereignty decision that is theirs, not ours |
| Vector store | Qdrant, per session | Needs persistence, metadata filtering (active/inactive) and deletion; FAISS provides none of the three |
| Tool surface | Two tools, shown conditionally | `PlayBookTools` already does this: `LoadPlayBook` only appears when playbooks exist. `SearchDocuments` follows the same pattern, so an empty index costs zero prompt tokens |
| Citations | `(filename, page, chunk_id)` + link | Requires fixing link resolution first: frontend `:5173`, backend `:3001`, no proxy, so `/files/<id>/view` 404s |

## 4. Not adopted

| Technique | Why not |
|---|---|
| GraphRAG, LightRAG, HippoRAG | Little benefit for single-hop questions, expensive indexing. *Is GraphRAG Needed?* also reports a retrieval–generation gap |
| Agentic RAG (Self-RAG, CRAG loops) | 5–8× cost; SAGE already has its own orchestration layer |
| RAPTOR | Summary tree over a large corpus; ours is small |
| Parent-child chunking | Contradicted by the ablation |
| HyDE | Hurt both benchmarks |
| `map_reduce`, `refine` | Lost on quality and cost |
| RLM, LogicRAG, AtomicRAG, SproutRAG | Dec 2025 – Jul 2026, still experimental |
| Docling | Existing extractors may suffice; add only if extraction is measured to be the bottleneck |

## 5. The baseline

`vector_storage.py`: FAISS `IndexFlatL2`, `ada-002`, 500-word window,
PyPDF2, dense-only with a fixed distance cut-off. Not in the public repo —
**must be obtained from GT-ARC**, or "validated improvements" cannot be
evidenced.

Four defects, all by reading and running the code:

1. **No session isolation.** `vector_storage = VectorStorage()` at module
   level — one index shared by every user and session.
2. **The cut-off likely destroys recall.** `IndexFlatL2` returns *squared* L2
   and ada-002 vectors are unit-norm, so `d < 0.3` is exactly `cos > 0.85`.
   Whether that is too strict on real data is a hypothesis to test — but if it
   holds, the baseline will show high precision and
   low recall, and a precision-only table would flatter it. Recall,
   budget-aware recall and answerable-rate are therefore mandatory.
3. **Chunk ids collide.** Dividing by `size` (500) while stepping by
   `size - overlap` (450) yields `0, 0, 1, …, 9, 9, 10` — a duplicate every
   ~10 chunks, so a citation can point at the wrong passage.
4. **No persistence, no deletion, no de-duplication**, and a synchronous
   embedding call inside an async backend.

### Ablation axes

Each row is one difference from the baseline, measured alone. Enabling
everything and reporting one improved number would not support the claim.

| # | Axis | Baseline | This design |
|---|---|---|---|
| A1 | Chunking | 500-word window | ~300 tokens, boundary-aware |
| A2 | Embedding | `ada-002` (2022) | current model |
| A3 | Retrieval | dense only | dense + BM25, RRF |
| A4 | Selection | fixed `cos > 0.85` | top-k under a token budget |
| A5 | Reranking | none | cross-encoder / LLM, adaptive |
| A6 | Context | raw chunks | contextual chunking |
| A7 | Query | verbatim | conversational rewrite |
| A8 | Formats | PDF only | pdf/docx/pptx/xlsx/csv/txt/md |
| A9 | Tool surface | n/a | +2 tools — does tool selection regress? |

A9 is the axis a generic RAG evaluation would omit, and the one GT-ARC is
most likely to ask about.

## 6. Evaluation

Method borrowed from the ablation study: cheap-to-expensive funnel, wide grids
on free retrieval metrics, paid generation only for finalists.

| Phase | Varies | Cost |
|---|---|---|
| 0 | Question set + gold passages | free |
| 1 | Chunker | free |
| 2 | Retriever | free |
| 3 | Reranker × corpus size, EN and DE separately | ~free |
| 4 | Contextual chunking, query rewrite | low |
| 5 | Finalists: generation + RAGAS | paid |
| 6 | Tool-selection regression on SAGE's own benchmark | paid |

Metrics: recall@k, budget-aware `token_recall@B` at B ∈ {500, 1000, 2000},
answerable-rate, then RAGAS faithfulness and answer relevancy on finalists.
Budget-aware metrics matter most because the binding constraint is the context
budget, not ranking position.

**Trap:** the retrieval code is deliberately forgiving — if reranking fails it
logs a warning and continues. Correct in production, dangerous in measurement,
because a silently disabled stage produces a row identical to the baseline and
invites the conclusion "this feature does nothing". The harness must detect
fallbacks and mark such rows invalid.

## 7. Cost model

Projected, to be replaced after the evaluation run. Recorded here so the
assumptions behind it can be challenged.

Assumptions: 15 documents ≈ 200k tokens ≈ 665 chunks; 60 questions;
`text-embedding-3-small` $0.02/M, `gpt-4o-mini` $0.15/$0.60,
`gpt-4o` $2.50/$10.00; ~1.33 tokens per word. Prices are from mid-2026 and must be
re-verified before running; latency figures are order-of-magnitude, not
benchmarks.

One-off evaluation: roughly **$5 for a clean pass, ~$16 with two re-runs**.
The shape matters more than the total: ~75% falls in Phase 5, the only phase
using the expensive generator, while Phases 1–4 together come to under $0.20.
That is the argument for the funnel — most design questions are answered
almost for free, and a mistake in the cheap phases costs nothing to redo.

Recurring, per document indexed:

| Document | Chunks | Embedding | + contextual | Indexing time |
|---|---|---|---|---|
| 5-page report | 11 | $0.0001 | $0.0014 | ~3 s |
| 30-page guide | 66 | $0.0004 | $0.0083 | ~19 s |
| 150-page manual | 332 | $0.0020 | $0.0418 | ~99 s |

Contextual chunking multiplies indexing cost ~20× but stays trivial in
absolute terms — four cents for a 150-page manual, paid once. **Latency is its
real cost**, which is why it runs in the background.

Recurring, per query:

| Component | Cost | Added latency |
|---|---|---|
| Query embedding | $0.00001 | ~150 ms |
| BM25 (local) | $0 | ~20 ms |
| Query rewrite | $0.00010 | ~400 ms |
| Cross-encoder rerank (CPU) | $0 | ~200 ms |
| *LLM rerank (alternative)* | $0.00096 | ~1.5 s |

At 1,000 queries/month: ~$0.11 with the cross-encoder, ~$1.07 with LLM
reranking. **Money is not the deciding factor; latency is.** Against a 2.6 s
baseline the cross-encoder adds ~0.8 s and LLM reranking ~2.1 s — paid on every
query whether or not it changes the answer. This is what the German-reranker
question actually turns on: the multilingual option must earn a second and a
half on German queries specifically, not merely help on average.

## 8. Open questions

1. **German compounds.** *Raumtemperaturregler* vs a query for
   *Temperaturregler* — no punctuation fix helps. BM25 is half of hybrid
   retrieval, so this is a first-class gap. Phase 3.
2. **Multilingual reranker.** The CPU-viable option is English-only; the
   multilingual one needs a GPU; the API option costs ~1.5 s per query.
   Phase 3 compares all three.
3. **Data sovereignty.** Embedding via OpenAI sends document text outside the
   EU. GT-ARC's decision, not ours — which is why the model is one env var.

## 9. Assumption this design rests on

Most of the tuning assumes documents are **reference material of meaningful
length** — manuals, guides, specifications. The task statement supports this
(*"upload or dump multiple documents"*), but SAGE's own sample prompts contain
no document-QA questions at all, so it remains an assumption.

If it is wrong — if real usage is short notes and one-page PDFs — then
reranking, contextual chunking and the weight given to BM25 all need
revisiting. Challenge this first.

## 10. Implementation order

Each commit independently green and reviewable.

1. `feat(files)` — multi-format extraction, multi-file and zip upload
2. `fix(frontend)` — resolve backend file links at render time
3. `chore(rag)` — Qdrant service, dependencies
4. `feat(tools)` — session-scoped context, collection cleanup
5. `feat(rag)` — chunking, embedding, indexing, search, two tools
6. `feat(rag)` — hybrid search, adaptive reranking
7. `feat(rag)` — conversational query rewrite
8. `test(benchmark)` — bilingual question set, ablation harness
9. `docs` — this document, with measured results replacing estimates

## References

TU Berlin Applied AI Project (SS 2026), *Advanced Techniques for Integration
of RAG and LLM* · GT-ARC, *SAGE RAG Improvement, Phase 1* (Jul 2026) ·
Anthropic, *Introducing Contextual Retrieval* (Sep 2024) · Wang et al.,
*Searching for Best Practices in RAG*, EMNLP 2024 · Es et al., *RAGAS*,
EACL 2024 · Chen et al., *Is GraphRAG Needed?*, arXiv:2606.25656
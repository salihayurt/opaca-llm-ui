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
| Chunking | ~500 tokens, boundary-aware | Set by measurement, not by the ablation. See 6b: the peak sits between 500 and 800 on all three documents, and 300 is below the useful range on every one |
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

## 6b. Measured results

Two documents, English and German GDPR, 17 and 7 usable questions. Retrieval
only, no generated answers. Small numbers: a difference of 0.05 here is
noise, and only the large and consistent movements are reported as findings.

### English, n=17

| Configuration | R@1 | R@3 | R@10 | MRR | answerable |
|---|---|---|---|---|---|
| baseline (as written) | 0.118 | 0.118 | 0.118 | 0.118 | **0.118** |
| baseline, threshold removed | 0.353 | 0.706 | 0.765 | 0.515 | 1.000 |
| hybrid, 150 tokens | 0.235 | 0.471 | 0.588 | 0.342 | 1.000 |
| hybrid, 300 tokens | 0.235 | 0.471 | 0.588 | 0.356 | 1.000 |
| hybrid, 500 tokens | 0.294 | 0.529 | 0.647 | 0.431 | 1.000 |
| **hybrid, 800 tokens** | **0.353** | 0.529 | **0.706** | **0.474** | 1.000 |
| dense only, 300 tokens | 0.294 | 0.353 | 0.588 | 0.366 | 1.000 |
| floor 0.20 / 0.30 | 0.235 | 0.471 | 0.588 | 0.356 | 1.000 |
| floor 0.40 | 0.235 | 0.412 | 0.588 | 0.335 | 1.000 |

### German, n=7

| Configuration | R@1 | R@3 | R@10 | MRR | answerable |
|---|---|---|---|---|---|
| baseline (as written) | 0.000 | 0.000 | 0.000 | 0.000 | **0.143** |
| baseline, threshold removed | 0.429 | 0.571 | 0.571 | 0.500 | 1.000 |
| hybrid, 150 tokens | 0.429 | 0.714 | 0.857 | 0.595 | 1.000 |
| hybrid, 300 tokens | 0.429 | 0.714 | 0.857 | 0.560 | 1.000 |
| hybrid, 500 tokens | 0.429 | **0.857** | 0.857 | 0.619 | 1.000 |
| **hybrid, 800 tokens** | 0.429 | **0.857** | 0.857 | **0.643** | 1.000 |
| dense only, 300 tokens | 0.286 | 0.429 | 0.714 | 0.398 | 1.000 |
| floor 0.20 / 0.30 / 0.40 | 0.429 | 0.714 | 0.857 | 0.560 | 1.000 |

### What the numbers settle

**The baseline's threshold is the finding, and it is worse than Section 5
supposed.** It returns nothing at all for 15 of 17 English questions and 6 of
7 German ones. In German it never once retrieves the right article: R@1
through R@10 are all zero. Removing the cut-off and changing nothing else
takes MRR from 0.118 to 0.515 in English and from 0.000 to 0.500 in German.
The mechanism is confirmed -- squared L2 below 0.3 is cosine above 0.85, and
relevant ada-002 pairs mostly sit below it.

This also fixes how the comparison should be stated. Against the baseline as
written, everything here looks transformative. Against the baseline with its
threshold removed, the honest claim is narrower and depends on the language:
in English the new pipeline is behind on recall (R@3 0.529 against 0.706) and
behind on MRR (0.474 against 0.515); in German it is ahead on both (0.857
against 0.571, MRR 0.643 against 0.500).

**Larger chunks win, in both languages, against the design and against the TU
Berlin ablation.** Section 4.1 committed to ~300 tokens on the strength of a
study where small chunks won everywhere. Here MRR rises monotonically with
chunk size in English (0.342, 0.356, 0.431, 0.474 at 150, 300, 500, 800) and
in German (0.595, 0.560, 0.619, 0.643). Two languages agreeing is harder to
dismiss than one document.

The likely reason is structural: GDPR articles are long, and a 300-token
window cuts one in half so neither piece carries the whole provision. That
makes it a property of this document rather than of retrieval in general, and
a maintenance manual with short procedural sections could well reverse it
again. The default stays at 300 for now, but it is now known to be wrong for
at least one realistic document type, and chunk size is the first axis to
sweep on any new corpus.

**Hybrid retrieval earns its place in German and not in English.** English:
R@3 0.471 hybrid against 0.353 dense, MRR 0.356 against 0.366 -- within
noise. German: R@3 0.714 against 0.429, MRR 0.560 against 0.398 -- a real
gap. The likely reason is compounds: a term like 'Kohaerenzverfahren' is rare
in embedding space and exact for BM25, which is the case hybrid retrieval
exists for. Hybrid stays on, and the German result is the evidence for it.

**The relevance floor does nothing.** 0.20 and 0.30 are identical to no floor
in both languages; 0.40 costs a little in English and nothing in German.
`min_dense_score` stays off, and the question moves from what value to set to
whether the mechanism is worth keeping at all.

### Third document: a maintenance manual, n=27

Run to test whether the chunk-size result was a property of the GDPR rather
than of retrieval. The document is the opposite shape -- short procedural
sections, code tables, a glossary, an FAQ -- and being a DOCX it has no
extraction damage, so all 27 questions verify.

| Configuration | R@1 | R@3 | R@10 | MRR | answerable |
|---|---|---|---|---|---|
| baseline (as written) | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** |
| baseline, threshold removed | 0.222 | 0.630 | 0.963 | 0.441 | 1.000 |
| hybrid, 100 tokens | 0.333 | 0.444 | 0.815 | 0.443 | 1.000 |
| hybrid, 150 tokens | 0.444 | 0.630 | 0.926 | 0.584 | 1.000 |
| hybrid, 300 tokens | 0.519 | 0.926 | 1.000 | 0.714 | 1.000 |
| **hybrid, 500 tokens** | 0.593 | 0.889 | 1.000 | **0.760** | 1.000 |
| hybrid, 800 tokens | 0.556 | **0.963** | 1.000 | 0.730 | 1.000 |
| hybrid, 1200 tokens | 0.519 | 0.815 | 1.000 | 0.693 | 1.000 |
| **dense only, 300 tokens** | **0.667** | 0.926 | 1.000 | **0.810** | 1.000 |
| floor 0.40 | 0.630 | 0.963 | 1.000 | 0.781 | 1.000 |

**The baseline retrieves nothing at all.** Not a low score: `answerable =
0.000`, no chunk returned for any of 27 questions. Two of its defects
compound here. Its 500-word windows put 20,000 characters into 8 chunks, each
covering three sections, so each embedding sits near the average of unrelated
material. Its threshold then demands cosine above 0.85, which such an average
never reaches against a short specific query. Removing the threshold alone
takes MRR from 0.000 to 0.441.

**The chunk-size hypothesis was wrong.** The GDPR result was explained as
long articles being cut in half by a 300-token window, which predicted the
effect would weaken on a document of short sections. It does not: MRR rises
0.443, 0.584, 0.714, 0.760 at 100, 150, 300, 500 tokens, peaks around 500 and
falls again by 1200. The same shape as the GDPR, on a document built the
opposite way.

Three documents now agree that 300 tokens is below the useful range and that
the peak sits somewhere around 500 to 800. The design committed to ~300 on
the strength of an ablation over different corpora, and every corpus measured
here disagrees. That is enough to change the default.

**Dense-only beats hybrid on this document**, MRR 0.810 against 0.714, and
the gap is entirely in the semantic category: 0.833 against 0.486. This is
the third different answer to the same question -- English GDPR called it a
tie, German favoured hybrid, this favours dense. The pattern that fits all
three is that BM25 helps where queries carry rare exact terms and hurts where
they are paraphrases, and which of those dominates is a property of the
questions users ask rather than of the pipeline. Hybrid stays on, since it is
never far behind and is well ahead on German compounds, but it is not the
free improvement the design assumed.

### What the numbers cannot settle

**The compound-word question is not answered.** All three surviving German
compound questions score 1.000 under every configuration including dense-only,
which says only that three questions were easy. Seven of the ten compound
questions were dropped in verification, and the ones that survived may well be
the ones the extraction damaged least. Section 8 stays open.

**Reranking is untested.** No cross-encoder is implemented, so it is absent
from the grid. The English results give it a target -- semantic questions
score R@3 0.250 against 0.750 for lexical, so ranking is where the loss is --
but choosing a reranker on 17 questions would be the same mistake as tuning
the extraction repair on three examples.

**Sample sizes are small enough to mislead.** Seventeen and seven questions,
one document each. The English and German sets are not translations of each
other, so a difference between the languages could be a difference between the
question sets. What survives that caveat is the direction of the chunk-size
effect, which is consistent across both, and the baseline threshold result,
which is too large to be noise.

### A note on how the extraction repair was settled

Worth recording because the process went wrong first. The repair was tuned
five times against handfuls of examples, and the measured result went 17
usable questions, then 12, then 11 -- worse at every step. Each round fixed
the example in front of it and broke something the examples did not cover: a
ceiling that stops 'direct or' becoming 'director' also refuses
'a ufsichtsbehoerden', which is the commonest shape of the defect in German.

Swept against both documents, the settings order clearly:

| Setting | English | German |
|---|---|---|
| no ceiling | 17/24 | 7/16 |
| ceiling 7.0 | 16/24 | 7/16 |
| ceiling 6.5 | 12/24 | 3/16 |
| ceiling 6.5, stronger document evidence | 11/24 | 2/16 |
| no repair at all | 1/24 | 0/16 |

The setting now in use is the first row, which is where the code was three
rounds of tuning ago. The wrong joins the ceiling prevents are real but
cheap: 'director indirect' damages one word and leaves the passage findable,
while a refused repair can remove a passage from reach entirely, because the
terms it would be found by no longer exist as tokens.

Raising the score given to document-supplied vocabulary, added on the
reasoning that German terms of art score zero in any frequency list, made
things worse in both languages and much worse in German. It let a term the
document happens to contain outrank the pieces of a join that should not
happen.

### What these numbers cannot say

Twelve questions on one document in one language. Confidence intervals on
proportions this small are wide enough that a 0.08 difference is noise. The
German set has not run at all, so the compound-word question in Section 8
remains entirely open. Reranking is absent from the grid because no
cross-encoder is implemented -- the decision to build one waits on a question
set where ranking quality is measurable, which this is not.

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
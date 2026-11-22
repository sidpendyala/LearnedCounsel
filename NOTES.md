# NOTES: problems, design changes, surprises

A running log; it becomes the lessons-learned section of the report.

## M0: environment (2026-11-21)
- Windows 11, RTX 4070 Laptop (8 GB), Ollama 0.40.17, Python 3.11.9 in `.venv`.
- `qwen3:8b` (default Q4_K_M) loaded with `num_ctx=8192` takes **6.2 GB, 100% GPU**, about 6.1 of 8.2 GB VRAM.
- `think: false` works: no thinking text is returned, and schema-constrained JSON came back valid on the first try.
- **Surprise:** the first call took 97 s (52 s load) and generated only 4.2 tok/s. It ran while another model was downloading. The warm repeat generated **42.6 tok/s** and took about 2 s per call. Benchmark only with the model warm and nothing else running.

## M1: data (2026-11-21)
- The spec expected `load_dataset` loading-script trouble. At the pinned commit, `nguha/legalbench` is plain TSVs with no script, so we read them directly at a pinned revision. This is simpler and fully reproducible.
- Text normalization barely matters: 5,307 unique posts by exact text, 5,299 after lowercase plus whitespace collapse, and the same with letters-and-digits only.
- One (post, task) pair has conflicting duplicate labels, so it is masked.
- 488 posts have 2+ known positives, so no hand-labeled set is needed.
- `education` (56 rows) and `benefits` (66) are the smallest tasks. N_A=25 per category fits but leaves little spare.
- The longest post is about 4,600 tokens. Prompts with definitions plus a long post could get close to `num_ctx`, which M2 has to check.

## M2: LLM client and speed benchmark (2026-11-21)
Results are in `results/benchmark_m2/`: `run1_open_list_schema.json`, `parallel.json` and `swap.json`.
- **Schema shape matters a lot for a small model.** A specialist schema with an open-ended `claims: [...]` list made qwen3:8b keep adding items until it hit the 1,024-token cap. Outputs averaged 1,362 tokens, and 6 of 8 calls needed a repair retry. With **one fixed field per category** plus `maxLength` on the strings, it averaged 188 tokens with 0 retries. Design rule: no open lists in agent schemas where a fixed set of keys will do.
- **Speed for qwen3:8b, warm:** about 7.5 s per B3-style call (1,088 tokens in, 227 out) and 6.7 s per specialist-style call (1,106 in, 188 out).
- **2 parallel requests:** on the same 16 calls, 80 s vs 114 s, so **1.42× throughput**. GPU stayed at 100%, context at 8,192, and the outputs were identical. Adopted concurrency 2, run across posts.
- **Thinking mode:** on 2 B3-style calls, think on took 17.9 s and 620 output tokens per call; think off took 6.7 s and 181 tokens. That's 2.7× slower, so think stays off everywhere.
- **qwen3:14b runs at only 58% GPU** (10.85 GB loaded) and takes 53 s per B3-style call, 7× slower than 8B. That isn't acceptable for the budget, so experiment (g) will use **qwen3:4b specialists vs qwen3:8b** (4B: 3.9 GB, 100% GPU, about 6 s per call).
- **8B↔4B swaps cost about 4–5 s each**, because both models can't stay loaded together. Mixed-model runs must use concurrency 1, since two posts in different phases would keep swapping the models back and forth.
- `prompt_eval_count` reports the full prompt even when Ollama reuses its prompt cache: 1,088 tokens in both serial and parallel runs. Token accounting is therefore consistent.
- **Sample sizes:** a rough per-post cost estimate for LearnedCounsel (about 11 calls) put N_A=25 plus everything else well over 12 h. Cut to N_A=15 (240 items), a 120-post Set B sample stratified 40/50/30 by number of positives, and a 60-post ablation subset. M4/M5 will replace the estimate with measured cost.

## M3: baselines B1 and B3 smoke test (2026-11-21)
- **B1 (TF-IDF + LR) is a strong baseline:** 5-fold CV mean accuracy **0.848**, with folds grouped by post so duplicates can't leak. It is *supervised* on about 80% of each task's own data, while the LLM systems are zero-shot, so the report must state that difference. On Set A items it gets 0.817 mean accuracy (out-of-fold predictions). Results: `results/b1_main/`.
- **B3 smoke test** (20 Set B posts, `results/smoke_b3/`): micro-F1 0.889 (P 0.865, R 0.914), exact match 0.600, 0 invalid outputs, 387 output tokens and 1 call per post, 9.2 s of wall time per post at concurrency 2. On the same 20 posts B1 gets micro-F1 0.943. With only 20 posts these numbers are not conclusive, just a sanity check.
- B3 outputs average 387 tokens, more than the 227-token benchmark call, because the fixed-field schema asks for 16 `present` and `evidence_quote` pairs.
- **Evidence check:** the first version marked Reddit HTML entities (`&gt;` written by the model as `>`) as mismatches. The fix decodes HTML entities before comparing. Even after that, about 12% of B3's quotes are paraphrased or invented ("I hit a parked car ... while I myself was parking" is not in the post). This is why LearnedCounsel re-prompts on invalid evidence.
- Re-running a finished experiment comes entirely from the cache (seconds, no model calls), so resuming works.

## M4: LearnedCounsel on 5 posts (2026-11-21)
Run: `results/m4_lc5/` (rendered transcripts are in `transcripts.txt`). 21 FakeLLM tests pass, covering the flow, τ enforcement, independence, the round cap and the ablation switches.
- **Cost:** 10.6 calls/post, 11.3k tokens in / 2.2k out, 0 invalid JSON messages, 0 coordinator failures. About **57 s of wall time per post** at concurrency 2, against the M2 estimate of 43 s. The M5 budget plan has to use the measured number.
- **The skeptic never made a challenge.** Specialists give positive claims confidence 1.00, so under τ=0.7 nothing is flagged. The skeptic did *try* to challenge (e.g. benefits, divorce), but the τ rule dropped those. The small model is overconfident, which makes τ nearly inert.
- **Possible misses are always "courts" and "torts", and specialists accept them every time.** All 8 possible misses in the 5 posts became present after the rebuttal, which looks like sycophancy. These categories are usually *masked* in Set B, so the masked metrics hide the extra false positives; Set A's negatives will show them.
- **A rebuttal contradicted itself:** its text said "not torts, it's breach of contract" while it returned present=true. The coordinator overruled it. The answer fields come before the reasoning field in the schema, so the model decides before it explains.
- **An evidence retry produced a valid but useless quote:** "I'm in Arkansas so I am well aware I have absolutely no rights at all." The verbatim check tests whether the words are in the post, not whether they're relevant.
- **Consult requests are frequent:** 1.2 per post, often loosely justified ("may relate to immigration or benefits"). They add about 1 specialist call per post.
- One of the 5 posts is from Manitoba, Canada, which falls outside the US taxonomy (a limitation for the report).
- On the known (unmasked) labels, all 5 posts are correct. With n=5 this means nothing statistically.

### M4 fixes and re-runs on the same 5 posts
Runs: `results/m4_lc5` (v1), `m4_lc5_v2`, `m4_lc5_v3` and `m4_lc5_v3_tau0.9`. Totals over the 5 posts:

| version | calls | consults | challenges kept | possible misses | flips | labels predicted | courts |
|---|---|---|---|---|---|---|---|
| v1 original | 53 | 6 | 0 | 8 | 10 | 17 | 4 |
| v2: reasoning-first schemas, calibrated confidence, stricter misses/rebuttal/consults | 53 | 5 | 0 | 3 | 2 | 19 | 5 |
| v3: v2 + official LegalBench definitions | 52 | 5 | 1 | 2 | 2 | 20 | 3 |
| v3 with τ=0.9 | 54 | 5 | 4 | 2 | 5 | 17 | 3 |

- **v2:** the sycophancy fix worked; flips dropped from 10 to 2. Confidences now use 0.8 for "implied" issues, but positive claims are never below 0.8, so **τ=0.7 still filters out every challenge**.
- **v3:** I had written `category_definitions.md` myself, and it disagreed with the official LegalBench/Learned Hands task descriptions. For example, officially *traffic* covers car purchases and repairs, *courts* covers hiring lawyers, *family* includes divorce and domestic violence, and *torts* includes bullying and neighbor disputes. The file is now rewritten from the official descriptions (github.com/HazyResearch/legalbench, `tasks/learned_hands_*/README.md`). The annotators labeled with those categories, so this is a validity fix that applies to all systems equally.
- **τ=0.9:** the skeptic becomes active. Its kept challenges removed `housing` (a known negative, so correct), `education` (the post only mentions a college savings plan) and `domestic_violence` on a landlord-tenant post. The cost was +2 calls over 5 posts. **Lesson:** τ only works if it sits above the model's typical "uncertain" confidence, and this model uses 0.8 for "implied" and almost never goes lower.
- Remaining issues: `courts` is still over-predicted on eviction and arrest posts, but it's masked for these posts, so we can't score it. `consumer` is added to housing deposit disputes; under the official definition ("money ... contracts") that may well be right.
- **Caution:** these prompt changes were developed by looking at 5 posts from the evaluation sample. To avoid tuning on test data, the 20 B3 smoke-test posts (which include these 5) should be treated as a dev set and excluded from final evaluation.

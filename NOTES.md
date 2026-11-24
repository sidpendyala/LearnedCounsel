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
- **2 parallel requests:** on the same 16 calls, 80 s vs 114 s, so **1.42× throughput**. GPU stayed at 100% and context at 8,192. The outputs were nearly but not exactly the same (specialist mean 188.5 vs 188.38 tokens), so temperature 0 is not bit-for-bit reproducible with 2 requests in flight. Adopted concurrency 2, run across posts.
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

- **v2:** flips dropped from 10 to 2 because the skeptic raised fewer possible misses. [CORRECTED in M5] That does not show sycophancy is fixed: in v3 specialists still accepted 2 of 2 possible misses. Confidences now use 0.8 for "implied" issues, but positive claims are never below 0.8, so **τ=0.7 still filters out every challenge**.
- **v3:** I had written `category_definitions.md` myself, and it disagreed with the official LegalBench/Learned Hands task descriptions. For example, officially *traffic* covers car purchases and repairs, *courts* covers hiring lawyers, *family* includes divorce and domestic violence, and *torts* includes bullying and neighbor disputes. The file is now rewritten from the official descriptions (github.com/HazyResearch/legalbench, `tasks/learned_hands_*/README.md`). The annotators labeled with those categories, so this is a validity fix that applies to all systems equally.
- **τ=0.9:** the skeptic becomes active. [CORRECTED in M5] Of 4 kept challenges, 1 flipped a label (`domestic_violence` on a landlord-tenant post). `housing` (a known negative) and `education` were removed by the coordinator's reopen questions; the specialist had kept `education` when the skeptic challenged it. The cost was +2 calls over 5 posts. **Lesson:** τ only works if it sits above the model's typical "uncertain" confidence, and this model uses 0.8 for "implied" and almost never goes lower.
- Remaining issues: `courts` is still over-predicted on eviction and arrest posts, but it's masked for these posts, so we can't score it. `consumer` is added to housing deposit disputes; under the official definition ("money ... contracts") that may well be right.
- **Caution:** these prompt changes were developed by looking at 5 posts from the evaluation sample. To avoid tuning on test data, the 20 B3 smoke-test posts (which include these 5) should be treated as a dev set and excluded from final evaluation.

## M5: pilot and runtime plan (started 2026-11-24)
- **Self-audit before M5.** Three M4/M2 statements were wrong or overstated and are corrected above, marked [CORRECTED in M5]. Cause: reading causes off summary counts instead of the debate log. From here on, flip causes are computed from the transcripts by code.
- **Dev set:** the 20 B3 smoke-test posts are in `data/dev_post_ids.txt` and excluded from Set A, the Set B sample and the ablation subset. All tuning (prompts, tau, B3 field order) uses only them.
- **Fairness:** the M4 fixes (evidence before decision, stricter "actual problem" wording) had been given to LearnedCounsel but not to B3. Both B3 versions are compared on dev and the better one is kept.
- `num_predict` raised from 1024 to 2048 for every system: 16-category outputs (B4, ablation d) could be cut off.
- **Battery power invalidates timings.** The first M5 dev run took 14 s per B3 post instead of about 9 s: the laptop was unplugged and the GPU was capped at 38 W / 1290 MHz (plugged in under load, M0: 86 W / 2415 MHz). Every run now records its power state and warns. `b3_present_first_dev` (and the start of `b3_evidence_first_dev`) ran on battery: their predictions are usable, their timings are not.
- **B3 version chosen on dev (20 posts, 54 known labels):** present_first micro-F1 0.930 [0.873, 0.984], evidence_first 0.904 [0.846, 0.958]. Kept present_first by the rule set beforehand. The intervals overlap heavily, so this is a default, not a finding; but "evidence before decision" gave no sign of helping B3, so it is not obviously a free win for LearnedCounsel either (M4 adopted it without testing it separately).
- B1 re-scored on the new evaluation samples (dev posts removed): `results/b1_main`. The B1 numbers in the M3 entry are superseded.

### M5 pilot on the 20 dev posts (54 known labels; tuning data, NOT reportable results)
Concurrency 2, AC power (except B3, whose cached calls were made on battery), 0 invalid outputs, 0 truncations, 0 context warnings. Source: `results/*_dev/`, `results/comparisons/learnedcounsel_dev.json`.

| system | micro-F1 | P | R | exact | calls/post | tokens/post | wall s/post |
|---|---|---|---|---|---|---|---|
| B3 | 0.930 | 0.917 | 0.943 | 0.750 | 1 | 1,532 | about 9.1 (from B5) |
| B4 | 0.919 | 0.872 | 0.971 | 0.700 | 1 | 2,623 | 28.2 |
| B5 (k=9) | 0.930 | 0.917 | 0.943 | 0.750 | 9 | 13,807 | 81.5 |
| LearnedCounsel (tau 0.9) | 0.925 | 0.969 | 0.886 | 0.800 | 10 | 12,412 | 52.9 |

- **tau = 0.9 chosen by the rule fixed beforehand:** micro-F1 0.925 vs 0.912 for 0.7 and 0.8, same 10.0 calls/post. A one-label difference, so it is a default, not a finding.
- **LearnedCounsel does not beat the baselines on dev.** Micro-F1 difference to B3 is -0.004 [-0.090, +0.077]; McNemar p = 1.0 against B3, B4 and B5, 0.73 against B2 (before Holm). It leans to precision (+0.097 over B4, CI excludes 0) and away from recall, the opposite of the hypothesis.
- **The router is the recall bottleneck.** Of 4 misses, 3 were in categories whose specialist was never called; the 4th was a correct round-0 claim that a coordinator reopen question talked the specialist out of. The prompt says to favor recall, but only 2.6 specialists per post run, consults included.
- **Sycophancy on possible misses is still there:** 6 of 6 accepted at tau 0.7, 4 of 5 at tau 0.9.
- **Challenges at tau 0.9:** 12 raised, 2 flipped a label, both on masked labels. The debate as a whole moved F1 from 0.909 (round 0) to 0.925 (final).
- **B5 equals B3 on all 320 labels.** Checked: sampling does vary (3.3 distinct texts per 9 samples), but in 50 of 56 (post, category) pairs with any yes vote all 9 samples agreed, so voting has almost nothing to change.
- **B4 invents or paraphrases many quotes:** 34 of 76 positive claims have a non-verbatim quote (evidence validity 0.55, vs 0.79 for B3 and 0.89 for LearnedCounsel, which re-prompts).
- B5 k = 9 is the nearest odd number to the token ratio 8.1; B5 then uses 1.11x LearnedCounsel's tokens.
- **Prompts are frozen here unless the user decides otherwise.** LearnedCounsel has already had several tuning rounds and B3 one, so more tuning would tilt the comparison. Ablation (b) measures the router's cost.
- **Runtime plan** (`python -m src.plan`, `results/runtime_plan.json`): about 16 h of new runs at the current sizes, 17.3 h with the 1.28 h already used in M5. The 12 h budget is not enough.

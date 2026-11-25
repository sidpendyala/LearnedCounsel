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

- **Pause (2026-11-24 18:33):** the main run was stopped by force at 247 saved transcripts (of 342 posts). The run had not written its ledger row, so `learnedcounsel_main_before_pause` was added from cache file timestamps. `metrics.json` of `learnedcounsel_main` will show cached calls and only the after-pause wall time; per-call latencies are the originals. Stopping the background task does not stop the Python process: it must be stopped separately.

## M6: full runs (interim, 2026-11-25 00:30; B5, ablations, latency and g still running)
- **LearnedCounsel does not beat B3 or B4.** Set B micro-F1 0.883 vs 0.909 (B3) and 0.908 (B4). Paired difference to B4: F1 -0.024 [-0.050, -0.000] (borderline, not corrected for multiple comparisons), recall -0.059 [-0.098, -0.021]. To B3: F1 -0.025 [-0.052, +0.000], recall -0.030 [-0.071, +0.013]. McNemar on Set A: p = 0.65 (B3), 0.63 (B4). Source: `results/comparisons/learnedcounsel_main.json`.
- **The pilot explanation "the router causes most misses" does not hold on the full Set B sample.** Of 39 misses: 17 the specialist looked and said absent, 15 the specialist was never called, 7 were present in round 0 and lost in the debate.
- **Giving in is not always bad.** Specialists accepted 53 of 68 possible misses (78%), and where the label is known 18 were correct and 2 incorrect. Skeptic challenges were the harmful part: 34 of 166 flipped a label, and of the 5 flips with a known label all 5 were incorrect (29 masked). Coordinator reopens: 16 correct, 15 incorrect, 81 masked.
- **The debate still helps a little on net** (Set B sample): F1 0.874 after round 0, 0.883 final. The weak point is the first pass, not the debate.
- **GPU heat:** at 22:28 the GPU was at 87-89 C with software thermal slowdown active (1605 MHz). It was only observed then, during B4, but B4 (and possibly other evening runs) may have inflated wall times; answers are not affected. At 00:33 it was back to 76 C with no slowdown.
- **One runaway reply** in 342 posts: intake of post 1b1376f93bc7 hit the 2048-token cap twice, so default intake notes were used. 4 prompt-length warnings, none over the limit.
- **B5 finished (02:49):** Set B micro-F1 0.907 [0.875, 0.934], differing from B3 on 10 of 1,920 labels while using about 9x its tokens. LearnedCounsel minus B5: F1 -0.023 [-0.049, +0.003].
- **To fix in `src/compare.py`:** B5 covers only 9 Set A items (posts shared with the Set B sample), so its McNemar test is meaningless and must be left out of the Holm family.
- **Second pause (2026-11-25 02:50):** stopped at the start of `learnedcounsel_no_skeptic_ablation`. One saved call was added to the ledger by hand. Resume with `bash scripts/run_all.sh`.
- **Few-shot (h) written (2026-11-25):** specialists see the shortest Yes and shortest No train post per category, in full, excluding every evaluation and dev post (`src/fewshot.py`, `prompts/few_shot.md`). Zero-shot prompts are unchanged: all 592 calls of the ablation baseline replay from the cache with the network disabled.
- **`src/compare.py` fixed:** Set A comparisons are skipped when the other run covers less than half of Set A (B5 covers 9 items by chance).
- **Safety finding (2026-11-25): some referral notes drift into legal advice**, although the coordinator prompt forbids it. Example (post 47e427): "Your father may be eligible for a B-2 visa ... He should contact the U.S. embassy". In 20 random notes of `learnedcounsel_main`, one reader (the AI assistant, not a lawyer) judged 3 as clear drifts (stating options in the law or possible arguments). No note is empty or over 120 words. A prompt rule alone does not enforce "no legal advice" in an 8B model: report as a limitation, keep the disclaimer prominent, tighten in v2.
- **Consult chains are cut at one level** (design choice), which can drop a useful request: in post 47e427 the consulted Work & Money specialist asked for Health & More (immigration), which was ignored; the skeptic's possible miss recovered immigration.

## M7/M8 work done while the queue ran (2026-11-25)
- `src/figures.py` draws all 8 figures from `results/` and prints TODO for runs that are missing. Looking at the images caught what tests could not: clipped boxes, a skeptic challenge that looked ignored (it had been dropped by the tau rule), and the legal-advice note above.
- `app.py` (Streamlit): only the presets use the disk cache; a visitor's own story is never cached and its transcript goes to a temporary folder. Streamlit's usage statistics are off (`.streamlit/config.toml`): with the default, "nothing left this computer" would have been false.
- The presets are written by us. The first version of preset 3 was too close to a dataset post (post 47e427) and was replaced. Whether the skeptic really changes its outcome still has to be verified with the model (TODO after the queue).
- matplotlib and streamlit only add packages to `.venv`; the versions used by the queue did not change (checked with a dry run first).

## M6 continued: ablations (2026-11-25)
- **Two runs failed (no_router, few_shot): the prompt-length guard was too strict.** It estimated 3.5 characters per token. Measured on 3,970 calls: median 4.51, minimum 4.09 for prompts of at least 8,000 characters; the longest prompt (20,766 characters) was 4,519 real tokens, not the estimated 5,933. The largest call used 5,522 of 8,192 tokens including the answer. The estimate is now 4.0 and the failed runs were restarted. No prompt and no cached result changed.
- Earlier I checked the longest *post* plus an assumed 4,000 characters of claims and debate, which underestimated the coordinator prompt.

### Ablation and optimization results (60-post subset of the Set B sample; paired bootstrap vs the full system; 10 comparisons, NOT corrected for multiple testing)
Source: `results/learnedcounsel_*_ablation/`, `figures/fig4_ablations.png`.

| variant | micro-F1 | change | recall change | tokens/post |
|---|---|---|---|---|
| full system | 0.877 | - | - | 12,911 |
| (a) no skeptic | 0.882 | +0.005 [-0.021, +0.034] | +0.009 | 11,874 |
| (b) no router (all 5 specialists) | 0.886 | +0.009 [-0.017, +0.038] | +0.043 [+0.000, +0.087] | 13,421 |
| (c) no consults | 0.872 | -0.005 [-0.016, +0.000] | -0.009 | 12,700 |
| (d) one general specialist | 0.834 | -0.043 [-0.089, +0.004] | -0.068 [-0.140, +0.008] | 10,027 |
| (e) 0 rounds | 0.845 | -0.032 [-0.064, +0.000] | -0.051 [-0.098, -0.008] | 7,712 |
| (e) 1 round | 0.850 | -0.026 [-0.056, +0.000] | -0.043 [-0.085, +0.000] | 10,097 |
| (f) tau 0.5 | 0.891 | +0.014 [+0.000, +0.033] | +0.017 | 12,533 |
| (f) tau 0.7 | 0.882 | +0.005 [-0.011, +0.025] | +0.009 | 12,446 |
| (g) 4B specialists | 0.816 | -0.061 [-0.111, -0.008] | -0.103 [-0.176, -0.018] | 12,773 |
| (h) few-shot | 0.869 | -0.008 [-0.048, +0.035] | +0.000 | 15,353 |

- **The router costs recall.** Calling all 5 specialists raises recall by 0.043 (the CI touches 0) and lowers precision by 0.031. With all 5 specialists the first pass already reaches F1 0.882, and the debate adds almost nothing (0.886).
- **The debate helps mainly when the router has left specialists out** (0.844 to 0.877 with the router), through the coordinator's reopen questions, not the skeptic.
- **The skeptic does not earn its cost:** removing it changes F1 by +0.005. The tau chosen on 20 dev posts (0.9) was not the best value on this subset; it stays at 0.9 for the reported system, because changing it now would be tuning on evaluation posts.
- **Diagnosis only (scored posts, not a result):** taking the union of LearnedCounsel and B3 would give F1 0.922 and recall 0.915 on these 60 posts, because the two make different mistakes. This motivates v2.
- **Latency, one request at a time, no cache, 5 dev posts:** B3 11.3 s, B4 29.2 s, LearnedCounsel 70.7 s, B5 100.9 s per post.
- 17.13 of 23 hours used.

## v2 (2026-11-25): design and rules fixed BEFORE running it
- **Design, from the first evaluation:** no router (b); a Scout (one broad call, the same as B3) that the specialists do not see; questions to the specialists where they disagree with it; a skeptic that raises possible misses only; a stricter referral-note rule. The original prompts and cached calls are unchanged (592 of 592 calls replay from the cache; evaluation set files byte-identical after the rebuild).
- **Data:** `v2_dev` (24 posts: 8 / 11 / 5 with 1 / 2 / 3+ known issues) and `v2_test` (90 posts: 30 / 42 / 18), disjoint from every post used so far and from the train split. Only 23 unused posts with 3+ issues were left, so the 3+ share is smaller than planned.
- **Candidates on v2_dev:** v2_add (the Scout may only add issues), v2_both (it may also question claims), v2_no_scout.
- **Selection rule:** best Set B micro-F1 on v2_dev; on a tie, fewer tokens. No prompt changes after that.
- **Test:** the chosen v2 runs ONCE on v2_test, against B3, B4, B1 and the original LearnedCounsel. Primary comparison: micro-F1 of v2 vs B3 (paired bootstrap); secondary: recall, vs B4, vs v1. It is reported whatever the result.
- **v2 candidates on v2_dev (24 posts, 69 known labels; tuning data, NOT reportable):** v2_both micro-F1 0.946 [0.899, 0.989] (P 0.917, R 0.978, 16,975 tokens/post), v2_add 0.926, v2_no_scout 0.925; B3 0.941 (P 1.000, R 0.889, 1,537 tokens/post). **v2_both is chosen by the rule; nothing is changed after this point.** The margin is a few labels, and v2 uses about 11x B3's tokens.
- On these 24 posts the Scout's questions flipped 7 labels; the 4 with a known label were all correct. Reopens flipped 2 known labels, both wrong.

### v2 test result (run once on 90 fresh posts, 262 known labels; 0 invalid outputs; AC power)
Source: `results/learnedcounsel_v2_test`, `learnedcounsel_v1_test`, `b3_v2_test`, `b4_v2_test`, `results/comparisons/learnedcounsel_v2_test.json`.

| system | micro-F1 | P | R | recall 3+ | labels/post | false positives on 92 known negatives | verbatim quotes | tokens/post |
|---|---|---|---|---|---|---|---|---|
| B3 | 0.906 [0.872, 0.939] | 0.967 | 0.853 | 0.768 | 2.67 | 5 | 0.908 | 1,449 |
| B4 | 0.920 [0.890, 0.947] | 0.928 | 0.912 | 0.875 | 3.42 | 12 | 0.620 | 2,528 |
| LearnedCounsel v1 | 0.911 [0.880, 0.943] | 0.949 | 0.876 | 0.804 | 2.96 | 8 | 0.958 | 11,959 |
| LearnedCounsel v2 | 0.924 [0.896, 0.950] | 0.919 | 0.929 | 0.911 | 3.42 | 14 | 0.948 | 15,279 |

- **Primary comparison (fixed beforehand), v2 vs B3 micro-F1: +0.018 [-0.007, +0.045], not significant.** Recall +0.076 [+0.036, +0.117], precision -0.048 [-0.079, -0.020]. McNemar p = 0.54.
- **v2 vs B4: no measurable difference** (F1 +0.004 [-0.017, +0.027], recall +0.018 [-0.013, +0.052]), at 6x B4's tokens. The clear gain over B4 is evidence: +0.328 verbatim quotes [+0.250, +0.406].
- **v2 vs v1:** recall +0.053 [+0.012, +0.091], precision -0.030 [-0.061, -0.001], F1 +0.013 [-0.014, +0.039].
- **Most of v2's recall gain is labelling more.** It outputs 3.42 labels per post against 2.67 for B3, and marks 14 of 92 known negatives against 5. The known labels are 170 positives to 92 negatives, so F1 here rewards generous systems. B4 outputs the same 3.42 labels per post. Do not present the recall gain without this.
- v1 vs B3 on fresh posts: F1 +0.005 [-0.025, +0.034]; on the first evaluation it was -0.025. The two samples disagree in sign: v1 and B3 are not distinguishable overall.
- The secondary comparisons are not corrected for multiple testing.
- 20.94 of 23 hours used.
- **Demo presets verified on the real model (2026-11-25).** Preset 3 (bakery): round 0 finds business, courts and housing; the skeptic's possible miss adds torts. The first "one issue" preset (a withheld security deposit) ended as housing + consumer + courts, because the coordinator's reopen questions talked specialists into two extra labels. It was replaced by an overtime story that gives employment only. The replacement was chosen because it gives a clear demo, so it is not evidence of quality.
- README.md and report/REPORT_DRAFT.md written from the saved results (2026-11-25). Case studies were read in the transcripts before being written up.

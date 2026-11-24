# Data Card: LearnedCounsel evaluation data

## Source
- **Dataset:** the 16 `learned_hands_*` tasks of **LegalBench** (Guha et al., NeurIPS 2023), which come from the **Learned Hands** project (Stanford Legal Design Lab and Suffolk LIT Lab).
- **Content:** anonymized Reddit (r/legaladvice) posts in which people describe legal problems. Lawyers and law students labeled each post with legal issue categories from the LIST taxonomy.
- **Hugging Face repo:** `nguha/legalbench`, pinned to commit `daec8237410aa23e3faf4bc41ad8b3a7e1696826` (repo last modified 2026-03-30).
- **Files:** `data/learned_hands_{category}/{train,test}.tsv`, read directly with pandas through `huggingface_hub.hf_hub_download`.
- **License:** CC BY-NC-SA 4.0 (non-commercial, share-alike, attribution required).
- **Rebuild:** `python -m src.data` writes everything into `data/processed/`, which git ignores.

## Format checks (verified against the spec)
| Claim in the spec | Verified? |
|---|---|
| Each task is binary and class-balanced | **Yes.** Every task's test split is exactly 50/50 Yes/No, except `family`, which has 1133 Yes and 1132 No. |
| The train split is tiny, meant for few-shot prompts | **Yes.** It has 6 rows per task (3 Yes, 3 No), 96 rows in total. |
| `load_dataset` may need an old loading script | **Doesn't apply.** At this commit the repo has no loading script, only TSV files, so we skip `load_dataset` and read the TSVs directly. |
| Column names | `index` (int, position within the task), `text` (the post), `answer` (`Yes`/`No`). We normalize `answer` to `label` ∈ {0, 1}. |

## Test rows per category
| category | rows | category | rows |
|---|---|---|---|
| benefits | 66 | estates | 178 |
| business | 174 | family | 2265 |
| consumer | 614 | health | 226 |
| courts | 192 | housing | 4494 |
| crime | 688 | immigration | 134 |
| divorce | 150 | torts | 432 |
| domestic_violence | 174 | traffic | 556 |
| education | 56 | employment | 710 |
| **total** | **11,109** | | |

The two smallest tasks are `education` (56 rows) and `benefits` (66 rows).

## Deduplication
- Posts are matched by **normalized text**: lowercased, with whitespace collapsed.
  - That yields 5,299 unique posts.
  - Exact-text matching gives 5,307, and stricter normalization that keeps only letters and digits also gives 5,299. The lowercase-plus-whitespace rule is therefore enough.
- `post_id` is the first 12 hex characters of the SHA-1 hash of the normalized text.
- 10 (post, task) pairs appear twice within the same task. One of them has conflicting labels (Yes in one row, No in the other), so that entry is **masked** in Set B.

## Set A (standard binary, comparable to LegalBench)
- **Size:** 15 items per category, stratified as 8 Yes and 7 No, with seed 448. That's **240 items** covering **230 unique posts**.
  - The spec started at 25 per category (400 items). That was cut to 15 after the M2 speed benchmark, to fit the 12-hour compute budget; see NOTES.md.
- **File:** `data/processed/set_a.csv`, with columns `item_id, post_id, category, text, label`.

## Set B (multi-label, masked)
- **Rows:** one per unique post (5,299), with a 16-dim label vector:
  - `1` means the post is a positive in that task.
  - `0` means it is a negative in that task.
  - Empty (masked) means the post never appears in that task.
- All multi-label metrics ignore masked entries.
- **File:** `data/processed/set_b.csv`, with columns `post_id, text, n_known, n_known_pos`, then one column per category.

| Number of tasks a post appears in | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 10 |
|---|---|---|---|---|---|---|---|---|
| posts | 1594 | 2156 | 1113 | 352 | 68 | 11 | 4 | 1 |

| Known positives per post | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| posts | 315 | 4496 | 421 | 59 | 7 | 1 |

- **488 posts have 2 or more known positives.** That is above the threshold of 100, so no hand-labeled set is needed.
- The 315 posts with 0 known positives appear only as negatives. They are probably positives for a LIST category outside our 16.

**Category pairs that are most often positive together:**

| pair | posts |
|---|---|
| divorce + family | 46 |
| domestic_violence + family | 44 |
| courts + crime | 38 |
| crime + torts | 28 |
| health + torts | 24 |
| consumer + housing | 22 |
| housing + torts | 22 |
| family + housing | 21 |

These line up with the confusable pairs in the spec (divorce/family, courts/crime, torts with others).

## Evaluation samples (sized to the compute budget)
- **Set B sample:** 120 posts, stratified by number of known positives so that multi-issue posts are oversampled: 40 with 1, 50 with 2, and 30 with 3 or more. Seed 448. File: `set_b_sample.csv`.
- **Ablation subset:** 60 posts drawn from the Set B sample (20 with 1, 25 with 2, 15 with 3 or more). The main run's cached calls are reused there. File: `ablation_subset.csv`.
- **Unique posts in the main comparison** (Set A posts plus Set B sample posts): **342**. Each post is run once per system, and both sets are scored from that one output.
- **Dev set:** 20 posts (`data/dev_post_ids.txt`) used to tune prompts, tau, the B3 version and the B5 k. They are excluded from Set A, the Set B sample and the ablation subset, and are never used for reported results. File: `dev.csv`.
- **Train/test overlap:** 82 posts of the train split also appear in the test split of some task, so few-shot examples must exclude evaluation posts.

## Post length
| | min | 25% | median | 75% | max |
|---|---|---|---|---|---|
| characters | 60 | 596 | 963 | 1644 | 18,318 |

- 186 posts are longer than 4,000 characters (about 1,000 tokens), 18 are longer than 8,000 characters, and 1 is longer than 16,000 characters (about 4,600 tokens).
- Every post fits within `num_ctx = 8192` on its own. The LLM client still checks each full prompt (definitions plus post).

## Known limitations
- **Reddit posts:** the sample is self-selected and informally written, with US-centric law.
- **Masked labels:** a missing label is **unknown**, not negative. Set B recall can only be measured on categories where the post's label is known, so a "multi-issue" post may have more issues than we know about.
- **Sampled negatives:** each task's negatives were randomly drawn from posts about other topics, so they are usually easy. Accuracy on the balanced binary tasks overstates real-world precision.
- **Overlap between sets:** Set A is drawn from the same test pool as Set B. That's intended: each unique post is run once and both sets are scored from the same output.
- **Comparability:** published LegalBench scores used much larger models, so our absolute numbers can't be compared with them directly.

## Citations
- Guha, N., et al. (2023). *LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models.* NeurIPS 2023 Datasets and Benchmarks Track.
- Learned Hands / LIST taxonomy: Stanford Legal Design Lab and Suffolk LIT Lab. (The exact reference will be verified for the report.)

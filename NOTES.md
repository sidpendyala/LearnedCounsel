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

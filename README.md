# LearnedCounsel: Multi-Agent Legal Issue Spotting for Access to Justice

CMPSC 448 final project, Penn State, Fall 2026
**Team leader:** Siddharth Pendyala. **Team member:** Palash Doshi.

LearnedCounsel reads a person's story and outputs (1) which of 16 legal issue categories it involves, each with quoted evidence, (2) an urgency level and (3) a short plain-language referral note. It is a team of AI agents (intake, router, five specialists, skeptic, coordinator) that share a case file, challenge each other and decide together. Everything runs on one laptop with a free local model.

> **This is not legal advice.** LearnedCounsel is a class project. It makes mistakes, and some of its notes drift into advice although the prompt forbids it. Talk to a lawyer or a legal aid organization about your situation.

**Name:** LearnedCounsel is named after the Learned Hands dataset (Stanford Legal Design Lab & Suffolk LIT Lab), itself a nod to Judge Learned Hand. It is unrelated to the commercial products Learned Hand AI or CoCounsel.

## What we found
- The agent team did **not** beat a single call to the same 8B model on our main metric, and it used about 8 times the tokens.
- An improved version (v2), tested once on 90 fresh posts, found more of the known issues than the single call (recall 0.929 vs 0.853), but it also labels more and is less precise (0.919 vs 0.967). Its micro-F1 advantage (+0.018) is not significant.
- A single prompt that runs the same workflow (B4) matched v2 at a sixth of the tokens, but only 62% of its quotes are really in the post, against 95% for v2.

Details: `report/REPORT_DRAFT.md`, figures in `figures/`, running log in `NOTES.md`.

## Setup (Windows, Git Bash; NVIDIA GPU with 8 GB)
1. Install [Ollama](https://ollama.com) (we used 0.40.17) and Python 3.11. No accounts, API keys or cloud models are used.
2. Run:
```
ollama pull qwen3:8b
ollama pull qwen3:4b                                   # only for experiment (g)
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m src.data                       # download LegalBench at a pinned commit, build the sets
.venv/Scripts/python -m pytest -q tests                # works without Ollama
```

## Reproduce each result
| Result | Command |
|---|---|
| Demo | `.venv/Scripts/python -m streamlit run app.py` |
| B1 (TF-IDF) | `.venv/Scripts/python -m src.experiments --system b1` |
| All other main runs, ablations, latency, (g), (h) | `bash scripts/run_all.sh` (about 16 h on our laptop) |
| v2: choose a candidate on dev posts | `bash scripts/run_v2_dev.sh` |
| v2: the single test on fresh posts | `bash scripts/run_v2_test.sh` |
| Main comparison statistics | `.venv/Scripts/python -m src.compare --main learnedcounsel_main --others b1_main b2_A b3_main b4_main b5_B` |
| v2 statistics | `.venv/Scripts/python -m src.compare --main learnedcounsel_v2_test --others b3_v2_test b4_v2_test learnedcounsel_v1_test b1_main --set v2_test` |
| Figures and the results table | `.venv/Scripts/python -m src.figures` |
| Read one transcript | `.venv/Scripts/python -m src.orchestrator logs/transcripts/<run>/<post_id>.jsonl` |

Runs are resumable (LLM calls are cached in `cache/`, which is not in git) and stop at `compute_budget_hours` in `config.yaml`. Keep the laptop plugged in: on battery the GPU is slowed down and timings are invalid. Temperature 0 with two parallel requests is not bit-for-bit reproducible, so a fresh run can differ in a few labels.

## Code map
| File | What it does |
|---|---|
| `src/data.py` | Loads LegalBench, builds Set A, Set B and the dev/test samples |
| `src/llm.py` | Ollama client: cache, JSON schema + one repair, runtime guard, context check; `FakeLLM` for tests |
| `src/schemas.py` | All structured outputs and the shared case file (pydantic) |
| `src/agents.py`, `src/orchestrator.py` | One function per agent; the control flow |
| `src/baselines.py` | B1 to B5 |
| `src/evaluate.py`, `src/stats.py`, `src/debate.py`, `src/compare.py` | Metrics; bootstrap and McNemar; what the debate did; system comparisons |
| `src/experiments.py` | Runs any system on any set and saves `results/<run>/` |
| `src/figures.py`, `app.py`, `prompts/` | Figures; demo; every prompt as editable Markdown |

## Data, license, privacy
Posts come from the Learned Hands tasks of LegalBench (Guha et al., 2023), license **CC BY-NC-SA 4.0**; see `data/DATA_CARD.md`. Files in `results/` and `figures/` contain short excerpts of these posts and are shared under the same license. The demo never writes a visitor's story to disk and turns off Streamlit's usage statistics. References: `report/REFERENCES.md`.

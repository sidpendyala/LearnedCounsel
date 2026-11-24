"""Runtime plan for the full runs, from pilot measurements (these are ESTIMATES, not results).

Run:  python -m src.plan        Writes results/runtime_plan.json
"""
import json
from pathlib import Path

import pandas as pd

from src.experiments import seconds_used

ledger = pd.read_csv("results/runtime_ledger.csv").set_index("run_name")
wall, calls = ledger["wall_seconds"], ledger["live_calls"]
debate = json.loads(Path("results/learnedcounsel_dev/metrics.json").read_text(encoding="utf-8"))
calls_per_post = debate["cost"]["calls_per_post"]

# Seconds of wall time per post at concurrency 2, plugged in.
LC = wall["learnedcounsel_tau0.7_dev"] / calls["learnedcounsel_tau0.7_dev"] * calls_per_post
B2_ITEM = wall["b2_dev"] / calls["b2_dev"]
B3 = wall["b5_dev"] / calls["b5_dev"]          # B5 is made of B3 calls (the B3 dev run was on battery)
B4 = wall["b4_dev"] / 20
B5 = wall["b5_dev"] / 20

# Ablations reuse cached calls of the main run, so only the changed calls cost time.
ABLATION = {
    "a no_skeptic (assumed)": 16, "b no_router (measured)": wall["learnedcounsel_no_router_dev_n5"] / 5,
    "c no_consults (assumed)": 10, "d single_specialist (measured)": wall["learnedcounsel_single_specialist_dev_n5"] / 5,
    "e rounds0 (assumed)": 8, "e rounds1 (assumed)": 8,
    "f tau0.5 (as tau0.9 rerun)": wall["learnedcounsel_tau0.9_dev"] / 20,
    "f tau0.7 (as tau0.9 rerun)": wall["learnedcounsel_tau0.9_dev"] / 20}
G = wall["learnedcounsel_mixed_4b_dev_n5"] / 5
H = LC * 1.15                                   # assumed: full cost plus longer prompts


def plan(main_posts, set_a_items, set_b_posts, ablation_posts, with_g_h):
    hours = {"LearnedCounsel main": main_posts * LC, "B2": set_a_items * B2_ITEM, "B3": main_posts * B3,
             "B4": main_posts * B4, "B5 (Set B only)": set_b_posts * B5,
             "latency check (5 posts, sequential, no cache)": 5 * (LC + B3 + B4 + B5) * 1.42,
             "ablations a-f": ablation_posts * sum(ABLATION.values())}
    if with_g_h:
        hours.update({"g mixed 4B": ablation_posts * G, "h few-shot": ablation_posts * H})
    hours = {k: round(float(v) / 3600, 2) for k, v in hours.items()}
    hours["TOTAL new runs"] = round(sum(hours.values()), 2)
    hours["TOTAL with hours already used"] = round(hours["TOTAL new runs"] + seconds_used() / 3600, 2)
    return hours


out = {"seconds_per_post": {"LC": LC, "B2 per item": B2_ITEM, "B3": B3, "B4": B4, "B5": B5, "g": G, "h": H, **ABLATION},
       "hours_used": seconds_used() / 3600,
       "current sizes (15 per category, 120 Set B, 60 ablation, g+h)": plan(342, 240, 120, 60, True),
       "fits 12 h (10 per category, 90 Set B, 45 ablation, no g/h)": plan(240, 160, 90, 45, False)}
out["seconds_per_post"] = {k: round(float(v), 1) for k, v in out["seconds_per_post"].items()}

if __name__ == "__main__":
    Path("results/runtime_plan.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

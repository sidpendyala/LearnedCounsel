"""What the agents' debate actually did, computed from the transcripts (not from summary counts).

Every change is scored against the known labels: correct, incorrect, or unknown (masked).
"""
import json
from pathlib import Path

import pandas as pd

from src.evaluate import binary_metrics
from src.schemas import CATEGORIES


def load_cases(folder, post_ids):
    cases = []
    for p in post_ids:
        lines = (Path(folder) / f"{p}.jsonl").read_text(encoding="utf-8").splitlines()
        cases.append(json.loads(lines[-1])["case"])
    return cases


def known_labels(set_b):
    """{(post_id, category): 0/1} for unmasked entries."""
    return {(r["post_id"], c): int(r[c]) for r in set_b.to_dict("records")
            for c in CATEGORIES if not pd.isna(r[c])}


def verdict(known, post_id, category, label):
    truth = known.get((post_id, category))
    if truth is None:
        return "unknown"
    return "correct" if bool(truth) == bool(label) else "incorrect"


def tally(verdicts):
    return {v: verdicts.count(v) for v in ["correct", "incorrect", "unknown"]}


def debate_metrics(cases, known):
    out, n = {}, len(cases)
    rebuttals = [(c["post_id"], d) for c in cases for d in c["debate"] if d["type"] == "rebuttal"]
    for kind in ["challenge", "possible_miss", "reopen"]:
        items = [(p, d) for p, d in rebuttals if d["kind"] == kind]
        flips = [(p, d) for p, d in items if d["before"] != d["after"]]
        out[kind] = {"count": len(items), "per_post": len(items) / n, "flips": len(flips),
                     "flip_rate": len(flips) / len(items) if items else None,
                     "flips_vs_known_labels": tally([verdict(known, p, d["category"], d["after"])
                                                     for p, d in flips])}
    out["skeptic_items_dropped"] = sum(d["dropped"] for c in cases for d in c["debate"]
                                       if d["type"] == "skeptic")

    overrides = [(c["post_id"], k, c["final"]["labels"][k]) for c in cases for k in CATEGORIES
                 if c["final"]["labels"][k] != c["claims"][k]["present"]]
    out["coordinator_overrides"] = {"count": len(overrides),
                                    "vs_known_labels": tally([verdict(known, *o) for o in overrides])}

    added = [(c["post_id"], k) for c in cases for q in c["consults"]
             for k in c["specialist_outputs"].get(q["specialist"], {})
             if k != "consult_requests" and c["final"]["labels"][k]]
    out["consults"] = {"count": sum(len(c["consults"]) for c in cases),
                       "posts_with_consult": sum(bool(c["consults"]) for c in cases),
                       "final_positives_in_consulted_areas": tally([verdict(known, p, k, True)
                                                                    for p, k in added])}

    # Did the debate help? Score three stages on the same known labels.
    stages = {
        "round0_specialists": lambda c, k: any(o.get(k, {}).get("present")
                                               for o in c["specialist_outputs"].values()),
        "after_debate": lambda c, k: c["claims"][k]["present"],
        "final_coordinator": lambda c, k: c["final"]["labels"][k]}
    by_post = {c["post_id"]: c for c in cases}
    keys = [(p, k) for (p, k) in known if p in by_post]
    out["stages"] = {name: binary_metrics([known[x] for x in keys],
                                          [int(bool(f(by_post[p], k))) for p, k in keys])
                     for name, f in stages.items()}

    conf = [round(v["confidence"], 2) for c in cases for o in c["specialist_outputs"].values()
            for k, v in o.items() if k != "consult_requests" and v["present"]]
    out["positive_claim_confidences"] = {str(v): conf.count(v) for v in sorted(set(conf))}
    return out

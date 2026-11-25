"""Compare one run against others: McNemar on Set A, paired bootstrap differences on Sets A and B.

Example:
  python -m src.compare --main learnedcounsel_main --others b1_main b2_A b3_main b4_main b5_B
  python -m src.compare --main learnedcounsel_dev --others b3_dev b4_dev --set dev

Differences are main minus other, on the posts BOTH runs covered.
Writes results/comparisons/<main>.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import load_config
from src.evaluate import score_set_a, score_set_b
from src.experiments import load_sets
from src.schemas import CATEGORIES
from src.stats import (holm, mcnemar, paired_difference, set_a_counts, set_a_metrics, set_b_counts,
                       set_b_metrics)


def shared(preds_1, preds_2):
    """Keep only the (post, category) pairs that both systems answered."""
    key = ["post_id", "category"]
    common = preds_1[key].merge(preds_2[key], on=key)
    return preds_1.merge(common, on=key), preds_2.merge(common, on=key)


def compare(main, other, set_a, set_b):
    p1, p2 = shared(main, other)
    out = {}
    # Skip Set A if the other run covers only a few of its items by chance (B5 runs on Set B only).
    if score_set_a(p1, set_a)["n_items"] >= 0.5 * len(set_a):
        out["mcnemar_set_a"] = mcnemar(p1, p2, set_a)
        out["set_a_difference"] = paired_difference(set_a_counts, set_a_metrics, p1, p2, set_a, CATEGORIES)
    if score_set_b(p1, set_b, CATEGORIES)["n_posts"]:
        out["set_b_posts"] = score_set_b(p1, set_b, CATEGORIES)["n_posts"]
        out["set_b_difference"] = paired_difference(set_b_counts, set_b_metrics, p1, p2, set_b, CATEGORIES)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--others", nargs="+", required=True)
    ap.add_argument("--set", default="eval", choices=["eval", "dev", "v2_dev", "v2_test"])
    args = ap.parse_args()
    sets = load_sets(load_config())
    set_a, set_b = ((sets["set_a"], sets["set_b_sample"]) if args.set == "eval"
                    else (sets[f"{args.set}_items"], sets[args.set]))
    load = lambda run: pd.read_csv(Path("results") / run / "predictions.csv")
    main_preds = load(args.main)
    # B2 answers only Set A items, so comparing it on Set B would be meaningless.
    results = {o: compare(main_preds, load(o), set_a, set_b if not o.startswith("b2") else set_b.iloc[:0])
               for o in args.others}
    p_values = {o: r["mcnemar_set_a"]["p_value"] for o, r in results.items() if "mcnemar_set_a" in r}
    for o, p in holm(p_values).items():
        results[o]["mcnemar_set_a"]["p_value_holm"] = p

    path = Path("results/comparisons") / f"{args.main}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for o, r in results.items():
        print(f"{args.main} - {o}")
        if "mcnemar_set_a" in r:
            m = r["mcnemar_set_a"]
            print(f"  Set A: only main right {m['only_1_correct']}, only other right {m['only_2_correct']} "
                  f"of {m['n_items']}; McNemar p={m['p_value']:.3f} (Holm {m['p_value_holm']:.3f})")
        for k in ["micro_f1", "micro_precision", "micro_recall"]:
            if "set_b_difference" in r:
                d = r["set_b_difference"][k]
                print(f"  Set B {k}: {d['value']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
                      f"{'  significant' if d['significant'] else ''}")
    print("Saved", path)


if __name__ == "__main__":
    main()

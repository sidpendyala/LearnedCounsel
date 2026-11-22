"""Run a system on a set of posts and save results. Resumable: LLM calls are cached on disk.

Examples:
  python -m src.experiments --system b1
  python -m src.experiments --system b3 --set B --n 20 --run-name smoke_b3

Sets: main (Set A posts + Set B sample), A, B (the Set B sample), ablation.
Writes results/<run_name>/ with metrics.json, predictions.csv, costs.csv, config.json.
"""
import argparse
import json
import platform
import subprocess
import threading
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd
import requests

from src.baselines import LLM_SYSTEMS, run_b1
from src.data import load_config
from src.evaluate import cost_summary, score_set_a, score_set_b
from src.llm import client_from_config
from src.orchestrator import Settings, learnedcounsel_post
from src.schemas import CATEGORIES


def load_sets(cfg):
    d = Path(cfg["data"]["processed_dir"])
    sets = {name: pd.read_csv(d / f"{name}.csv")
            for name in ["set_a", "set_b_sample", "ablation_subset", "test_long"]}
    for name in ["set_b_sample", "ablation_subset"]:
        sets[name][CATEGORIES] = sets[name][CATEGORIES].astype("Int8")
    return sets


def select_posts(sets, which, n, seed):
    """Unique posts (post_id, text) to run."""
    a = sets["set_a"][["post_id", "text"]]
    b = sets["set_b_sample"][["post_id", "text"]]
    pools = {"main": pd.concat([a, b]), "A": a, "B": b, "ablation": sets["ablation_subset"][["post_id", "text"]]}
    posts = pools[which].drop_duplicates("post_id").reset_index(drop=True)
    if n:
        posts = posts.sample(n=min(n, len(posts)), random_state=seed).reset_index(drop=True)
    return posts


def snapshot(cfg, args, model):
    """Everything needed to reproduce the run: config, exact model digest, hardware, git commit."""
    def sh(cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    snap = {"args": vars(args), "config": cfg, "time": datetime.now().isoformat(timespec="seconds"),
            "git_commit": sh(["git", "rev-parse", "--short", "HEAD"]),
            "hardware": {"gpu": sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                    "--format=csv,noheader"]),
                         "cpu": platform.processor(), "os": platform.platform()}}
    if model:
        tags = requests.get(cfg["ollama"]["url"] + "/api/tags", timeout=10).json()["models"]
        info = next(m for m in tags if m["name"] == model)
        snap["model"] = {"tag": model, "digest": info["digest"],
                         "quantization": info["details"].get("quantization_level"),
                         "parameters": info["details"].get("parameter_size"),
                         "ollama_version": requests.get(cfg["ollama"]["url"] + "/api/version").json()["version"]}
    return snap


def run_llm_system(cfg, system_fn, posts, budget_hours):
    llm = client_from_config(cfg, max_seconds=budget_hours * 3600 if budget_hours else None)
    done, lock, start = [0], threading.Lock(), time.time()

    def one(row):
        out = system_fn(llm, row.post_id, row.text)
        with lock:
            done[0] += 1
            if done[0] % 5 == 0 or done[0] == len(posts):
                print(f"  {done[0]}/{len(posts)} posts, {time.time() - start:.0f}s elapsed", flush=True)
        return out

    results = llm.map(one, list(posts.itertuples()), workers=cfg["ollama"]["concurrency"])
    preds = pd.DataFrame([r for rows, _ in results for r in rows])
    costs = pd.DataFrame([c for _, c in results])
    return preds, costs, llm.stats


# LearnedCounsel variants for the ablations (a)-(e) and optimization sweeps (f)-(g).
VARIANTS = {
    "full": {},
    "no_skeptic": {"use_skeptic": False},                 # (a)
    "no_router": {"use_router": False},                   # (b)
    "no_consults": {"use_consults": False},               # (c)
    "single_specialist": {"single_specialist": True},     # (d)
    "rounds0": {"max_rounds": 0},                         # (e)
    "rounds1": {"max_rounds": 1},                         # (e)
    "tau0.5": {"tau": 0.5},                               # (f)
    "tau0.9": {"tau": 0.9},                               # (f)
    "mixed_4b": {"specialist_model": "small"},            # (g) small model for specialists + rebuttals
}


def llm_systems(cfg, variant="full"):
    lc = cfg["learnedcounsel"]
    params = {"tau": lc["tau"], "max_rounds": lc["max_rounds"], **VARIANTS[variant]}  # variant wins
    if params.get("specialist_model") == "small":
        params["specialist_model"] = cfg["models"]["small"]
    settings = Settings(variant=variant, **params)
    return {**LLM_SYSTEMS, "learnedcounsel": partial(learnedcounsel_post, settings=settings)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["b1", "learnedcounsel"] + list(LLM_SYSTEMS))
    ap.add_argument("--set", default="main", choices=["main", "A", "B", "ablation"])
    ap.add_argument("--variant", default="full", choices=list(VARIANTS), help="LearnedCounsel variant")
    ap.add_argument("--n", type=int, default=None, help="random subset of n posts (smoke tests)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--budget-hours", type=float, default=None, help="runtime guard for live model time")
    args = ap.parse_args()

    cfg = load_config()
    sets = load_sets(cfg)
    variant = f"_{args.variant}" if args.system == "learnedcounsel" and args.variant != "full" else ""
    run_name = args.run_name or f"{args.system}{variant}_{args.set}" + (f"_n{args.n}" if args.n else "")
    out = Path("results") / run_name
    out.mkdir(parents=True, exist_ok=True)

    metrics = {"system": args.system, "variant": args.variant, "run_name": run_name}
    if args.system == "b1":
        preds, fold_acc = run_b1(sets["test_long"], cfg["seed"])
        metrics["cv_accuracy"] = {c: {"mean": float(pd.Series(a).mean()), "std": float(pd.Series(a).std())}
                                  for c, a in fold_acc.items()}
        metrics["cv_mean_accuracy"] = float(pd.Series([v["mean"] for v in metrics["cv_accuracy"].values()]).mean())
        model = None
    else:
        posts = select_posts(sets, args.set, args.n, cfg["seed"])
        print(f"Running {args.system} on {len(posts)} posts ({args.set})")
        preds, costs, stats = run_llm_system(cfg, llm_systems(cfg, args.variant)[args.system], posts, args.budget_hours)
        costs.to_csv(out / "costs.csv", index=False)
        metrics["cost"] = cost_summary(costs)
        extra = costs.drop(columns=["post_id", "calls", "tokens_in", "tokens_out", "latency_s", "valid"])
        metrics["per_post_means"] = extra.mean(numeric_only=True).to_dict()  # system-specific counters
        metrics["llm_stats"] = stats
        metrics["invalid_outputs"] = int((~costs["valid"]).sum())
        if "evidence_valid" in preds:
            ev = preds.loc[preds["pred"] == 1, "evidence_valid"]
            metrics["evidence_validity_rate"] = float(ev.mean()) if len(ev) else None
        model = cfg["models"]["main"]

    metrics["set_a"] = score_set_a(preds, sets["set_a"])
    metrics["set_b"] = score_set_b(preds, sets["set_b_sample"], CATEGORIES)
    preds.to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (out / "config.json").write_text(json.dumps(snapshot(cfg, args, model), indent=2))

    a, b = metrics["set_a"], metrics["set_b"]
    print(f"Saved {out}")
    if a["n_items"]:
        print(f"Set A: {a['n_items']} items, mean accuracy {a['mean_accuracy']:.3f}")
    print(f"Set B: {b['n_posts']} posts, micro-F1 {b['micro_f1']:.3f}, macro-F1 {b['macro_f1']:.3f}, "
          f"micro-P {b['micro_precision']:.3f}, micro-R {b['micro_recall']:.3f}, exact {b['exact_match']:.3f}")
    print("Recall by known positives:", {k: round(v, 3) for k, v in b["recall_by_known_positives"].items()})
    if "cost" in metrics:
        c = metrics["cost"]
        print(f"Cost/post: {c['calls_per_post']:.2f} calls, {c['tokens_in_per_post']:.0f} in, "
              f"{c['tokens_out_per_post']:.0f} out, {c['latency_s_per_post']:.1f}s; "
              f"invalid outputs {metrics['invalid_outputs']}, evidence validity "
              f"{metrics.get('evidence_validity_rate')}")


if __name__ == "__main__":
    main()

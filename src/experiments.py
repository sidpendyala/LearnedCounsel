"""Run a system on a set of posts and save results. Resumable: LLM calls are cached on disk.

Examples:
  python -m src.experiments --system b1
  python -m src.experiments --system b3 --set dev
  python -m src.experiments --system learnedcounsel --variant tau0.9 --set ablation
  python -m src.experiments --system b3 --set dev --n 10 --concurrency 1 --no-cache   (true latency)

Sets: main (Set A posts + Set B sample), A, B (the Set B sample), ablation, dev and v2_dev (tuning only),
v2_test (fresh posts for v2).
Writes results/<run_name>/ with metrics.json, predictions.csv, costs.csv, config.json.

Budget: every run adds its model runtime (wall clock) to results/runtime_ledger.csv, and
refuses new LLM calls once the total reaches compute_budget_hours in config.yaml.
"""
import argparse
import ctypes
import json
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd
import requests

from src.baselines import LLM_BASELINES, llm_baselines, run_b1
from src.data import load_config
from src.debate import debate_metrics, known_labels, load_cases
from src.evaluate import cost_summary, known_labels_long, score_set_a, score_set_b
from src.llm import RuntimeBudgetExceeded, client_from_config
from src.orchestrator import TRANSCRIPT_DIR, Settings, learnedcounsel_post
from src.schemas import CATEGORIES
from src.stats import set_a_counts, set_a_metrics, set_b_counts, set_b_metrics, with_ci

LEDGER = Path("results/runtime_ledger.csv")

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
    "tau0.7": {"tau": 0.7},                               # (f)
    "tau0.8": {"tau": 0.8},                               # (f), dev only
    "tau0.9": {"tau": 0.9},                               # (f)
    "mixed_4b": {"specialist_model": "small"},            # (g) small model for specialists + rebuttals
    "few_shot": {"few_shot": True},                       # (h)
    # v2 candidates (no router, skeptic raises misses only, stricter note); one is chosen on v2_dev
    "v2_add": {"use_router": False, "skeptic_misses_only": True, "safe_note": True, "scout_mode": "add"},
    "v2_both": {"use_router": False, "skeptic_misses_only": True, "safe_note": True, "scout_mode": "both"},
    "v2_no_scout": {"use_router": False, "skeptic_misses_only": True, "safe_note": True},
}


# ---------------- data ----------------
def load_sets(cfg):
    d = Path(cfg["data"]["processed_dir"])
    names = ["set_a", "set_b_sample", "ablation_subset", "dev", "v2_dev", "v2_test", "set_b", "test_long"]
    sets = {name: pd.read_csv(d / f"{name}.csv") for name in names}
    for name in ["set_b_sample", "ablation_subset", "dev", "v2_dev", "v2_test", "set_b"]:
        sets[name][CATEGORIES] = sets[name][CATEGORIES].astype("Int8")
    for name in ["dev", "v2_dev", "v2_test"]:   # these have no Set A: use their known labels as Yes/No items
        sets[f"{name}_items"] = known_labels_long(sets[name], CATEGORIES)
    return sets


def select_posts(sets, which, n, seed):
    """Unique posts (post_id, text) to run."""
    cols = ["post_id", "text"]
    pools = {"main": pd.concat([sets["set_a"][cols], sets["set_b_sample"][cols]]),
             "A": sets["set_a"][cols], "B": sets["set_b_sample"][cols],
             "ablation": sets["ablation_subset"][cols], "dev": sets["dev"][cols],
             "v2_dev": sets["v2_dev"][cols], "v2_test": sets["v2_test"][cols]}
    posts = pools[which].drop_duplicates("post_id").reset_index(drop=True)
    if n:
        posts = posts.sample(n=min(n, len(posts)), random_state=seed).reset_index(drop=True)
    return posts


# ---------------- runtime ledger ----------------
def seconds_used():
    return float(pd.read_csv(LEDGER)["wall_seconds"].sum()) if LEDGER.exists() else 0.0


def add_to_ledger(run_name, llm):
    if llm.wall_seconds() < 1:
        return  # everything came from the cache
    row = pd.DataFrame([{"time": datetime.now().isoformat(timespec="seconds"), "run_name": run_name,
                         "wall_seconds": round(llm.wall_seconds(), 1),
                         "live_calls": llm.stats["calls"] - llm.stats["cached_calls"]}])
    row.to_csv(LEDGER, mode="a", header=not LEDGER.exists(), index=False)


def keep_awake():
    """Stop Windows from sleeping while this process runs (it resets when the process ends)."""
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def on_battery():
    """True if the laptop is unplugged (the GPU is then power-capped, so timings are invalid)."""
    if sys.platform != "win32":
        return None
    status = (ctypes.c_byte * 12)()          # SYSTEM_POWER_STATUS; byte 0 = ACLineStatus (0 = battery)
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))
    return status[0] == 0


# ---------------- running ----------------
def snapshot(cfg, args):
    """Everything needed to reproduce the run: config, exact model digests, hardware, git commit."""
    def sh(cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    snap = {"args": vars(args), "config": cfg, "time": datetime.now().isoformat(timespec="seconds"),
            "git_commit": sh(["git", "rev-parse", "--short", "HEAD"]),
            "uncommitted_changes": bool(sh(["git", "status", "--porcelain", "src", "prompts", "config.yaml"])),
            "hardware": {"gpu": sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                    "--format=csv,noheader"]),
                         "cpu": platform.processor(), "os": platform.platform()}}
    if args.system != "b1":
        url = cfg["ollama"]["url"]
        tags = requests.get(url + "/api/tags", timeout=10).json()["models"]
        snap["models"] = {m["name"]: {"digest": m["digest"], "quantization": m["details"].get("quantization_level"),
                                      "parameters": m["details"].get("parameter_size")}
                          for m in tags if m["name"] in cfg["models"].values()}
        snap["ollama_version"] = requests.get(url + "/api/version", timeout=10).json()["version"]
    return snap


def llm_systems(cfg, variant, run_name, asked):
    lc = cfg["learnedcounsel"]
    params = {"tau": lc["tau"], "max_rounds": lc["max_rounds"], **VARIANTS[variant]}  # variant wins
    if params.get("specialist_model") == "small":
        params["specialist_model"] = cfg["models"]["small"]
    settings = Settings(variant=run_name, **params)   # transcripts go to logs/transcripts/<run_name>/
    return {**llm_baselines(cfg, asked), "learnedcounsel": partial(learnedcounsel_post, settings=settings)}


def run_llm_system(llm, system_fn, posts, workers):
    done, lock, start = [0], threading.Lock(), time.time()

    def one(row):
        out = system_fn(llm, row.post_id, row.text)
        with lock:
            done[0] += 1
            if done[0] % 5 == 0 or done[0] == len(posts):
                print(f"  {done[0]}/{len(posts)} posts, {time.time() - start:.0f}s elapsed", flush=True)
        return out

    results = llm.map(one, list(posts.itertuples()), workers=workers)
    return pd.DataFrame([r for rows, _ in results for r in rows]), pd.DataFrame([c for _, c in results])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["b1", "learnedcounsel"] + LLM_BASELINES)
    ap.add_argument("--set", default="main", choices=["main", "A", "B", "ablation", "dev", "v2_dev", "v2_test"])
    ap.add_argument("--variant", default="full", choices=list(VARIANTS), help="LearnedCounsel variant")
    ap.add_argument("--n", type=int, default=None, help="random subset of n posts")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--concurrency", type=int, default=None, help="default: config.yaml")
    ap.add_argument("--no-cache", action="store_true", help="re-run every call (for latency measurements)")
    args = ap.parse_args()

    cfg = load_config()
    sets = load_sets(cfg)
    variant = f"_{args.variant}" if args.system == "learnedcounsel" and args.variant != "full" else ""
    run_name = args.run_name or f"{args.system}{variant}_{args.set}" + (f"_n{args.n}" if args.n else "")
    out = Path("results") / run_name
    out.mkdir(parents=True, exist_ok=True)
    own_labels = args.set in ["dev", "v2_dev", "v2_test"]
    set_a = sets[f"{args.set}_items"] if own_labels else sets["set_a"]
    set_b = sets[args.set] if own_labels else sets["set_b_sample"]
    metrics = {"system": args.system, "variant": args.variant, "run_name": run_name, "set": args.set}

    if args.system == "b1":
        preds, fold_acc = run_b1(sets["test_long"], cfg["seed"])
        metrics["cv_accuracy"] = {c: {"mean": float(pd.Series(a).mean()), "std": float(pd.Series(a).std())}
                                  for c, a in fold_acc.items()}
        metrics["cv_mean_accuracy"] = float(pd.Series([v["mean"] for v in metrics["cv_accuracy"].values()]).mean())
    else:
        posts = select_posts(sets, args.set, args.n, cfg["seed"])
        asked = set_a.groupby("post_id")["category"].unique().to_dict()
        workers = args.concurrency or cfg["ollama"]["concurrency"]
        if args.variant == "mixed_4b":
            workers = 1  # two posts in different phases would keep swapping the models
        remaining = cfg["compute_budget_hours"] * 3600 - seconds_used()
        llm = client_from_config(cfg, max_seconds=remaining, cache_dir=None if args.no_cache else "cache/llm")
        print(f"Running {args.system}{variant} on {len(posts)} posts ({args.set}), {workers} at a time; "
              f"{remaining / 3600:.1f} h of compute budget left", flush=True)
        keep_awake()
        started_on_battery = on_battery()
        if started_on_battery:
            print("WARNING: running on battery, so the GPU is slowed down. Timings will not be valid.", flush=True)
        try:
            preds, costs = run_llm_system(llm, llm_systems(cfg, args.variant, run_name, asked)[args.system],
                                          posts, workers)
        except RuntimeBudgetExceeded as e:
            sys.exit(f"STOPPED: {e}")
        finally:
            add_to_ledger(run_name, llm)

        costs.to_csv(out / "costs.csv", index=False)
        metrics["cost"] = cost_summary(costs)
        metrics["concurrency"] = workers
        metrics["on_battery"] = {"start": started_on_battery, "end": on_battery()}
        metrics["wall_seconds_live"] = llm.wall_seconds()
        extra = costs.drop(columns=["post_id", "calls", "tokens_in", "tokens_out", "latency_s", "valid"])
        metrics["per_post_means"] = extra.mean(numeric_only=True).to_dict()
        metrics["llm_stats"] = llm.stats
        metrics["posts_with_invalid_output"] = int((~costs["valid"]).sum())
        if "evidence_valid" in preds:
            ev = preds.loc[preds["pred"] == 1, "evidence_valid"].dropna()
            metrics["evidence_validity_rate"] = float(ev.astype(bool).mean()) if len(ev) else None
        if args.system == "learnedcounsel":
            cases = load_cases(TRANSCRIPT_DIR / run_name, posts["post_id"])
            metrics["debate"] = debate_metrics(cases, known_labels(sets["set_b"]))

    metrics["set_a"] = score_set_a(preds, set_a)
    metrics["set_b"] = score_set_b(preds, set_b, CATEGORIES) if args.system != "b2" else None  # B2: Set A only
    metrics["ci"] = {"set_a": with_ci(set_a_counts, set_a_metrics, preds, set_a, CATEGORIES),
                     "set_b": with_ci(set_b_counts, set_b_metrics, preds, set_b, CATEGORIES)
                     if args.system != "b2" else None}
    preds.to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(snapshot(cfg, args), indent=2), encoding="utf-8")
    report(metrics, out)


def report(m, out):
    print(f"Saved {out}")
    if m["set_a"]["n_items"]:
        print(f"  Set A ({m['set_a']['n_items']} items): mean accuracy {m['set_a']['mean_accuracy']:.3f}")
    if m["set_b"]:
        b, ci = m["set_b"], m["ci"]["set_b"]["micro_f1"]["ci95"]
        print(f"  Set B ({b['n_posts']} posts, {b['n_known_entries']} labels): micro-F1 {b['micro_f1']:.3f} "
              f"[{ci[0]:.3f}, {ci[1]:.3f}], P {b['micro_precision']:.3f}, R {b['micro_recall']:.3f}, "
              f"macro-F1 {b['macro_f1']:.3f}, exact {b['exact_match']:.3f}")
    if "cost" in m:
        c, s = m["cost"], m["llm_stats"]
        print(f"  Per post: {c['calls_per_post']:.1f} calls, {c['tokens_in_per_post']:.0f} in / "
              f"{c['tokens_out_per_post']:.0f} out; this run {m['wall_seconds_live']:.0f}s of model time, "
              f"{s['cached_calls']}/{s['calls']} calls cached; invalid posts {m['posts_with_invalid_output']}, "
              f"repairs {s['repair_retries']}, evidence validity {m.get('evidence_validity_rate')}")


if __name__ == "__main__":
    main()

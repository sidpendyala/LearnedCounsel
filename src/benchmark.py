"""M2 speed benchmark.

Parts (python -m src.benchmark --part <name>):
  parallel  1 vs 2 requests in flight, on the SAME posts (parallel first, so Ollama's
            prompt cache cannot favor it); polls `ollama ps` for the GPU share
  models    thinking on vs off, and the small / strong models
  swap      cost of switching between the main and small model (they don't both fit in VRAM)
Each part writes results/benchmark_m2/<part>.json
"""
import argparse
import json
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
from pydantic import BaseModel, Field, create_model

from src.data import load_config
from src.llm import LLMClient

cfg = load_config()
CATS = cfg["categories"]
URL = cfg["ollama"]["url"]
DEFS = Path("prompts/category_definitions.md").read_text(encoding="utf-8")
OUT = Path("results/benchmark_m2")

# Two call shapes that resemble the real workload (the real prompts come in M3/M4).
FullLabels = create_model("FullLabels", **{c: (bool, ...) for c in CATS})


class Evidence(BaseModel):
    category: Literal[tuple(CATS)]
    evidence_quote: str = Field(max_length=300)


class FullOutput(BaseModel):
    labels: FullLabels
    evidence: list[Evidence]


class Claim(BaseModel):
    present: bool
    confidence: float = Field(ge=0, le=1)
    evidence_quote: str = Field(max_length=300)
    rationale: str = Field(max_length=200)


class SpecialistOutput(BaseModel):
    """One fixed field per category: an open list let the model generate claims forever (run 1)."""
    employment: Claim
    consumer: Claim
    benefits: Claim
    business: Claim


FULL_SYS = ("You read a person's description of a problem and decide which legal issue categories it "
            "involves. For every category answer true or false, then give one short verbatim quote "
            "from the post as evidence for each true category.\n\n" + DEFS)
SPEC_SYS = ("You are the Work & Money specialist. For each of your categories (employment, consumer, "
            "benefits, business) decide if the post involves it, with a confidence from 0 to 1, a short "
            "verbatim quote from the post as evidence, and a rationale of at most 25 words.\n\n" + DEFS)
CALL_TYPES = {"full": (FULL_SYS, FullOutput), "specialist": (SPEC_SYS, SpecialistOutput)}


def make_client():
    return LLMClient(URL, cfg["models"]["main"], num_ctx=cfg["ollama"]["num_ctx"],
                     num_predict=cfg["ollama"]["num_predict"], seed=cfg["seed"], cache_dir=None)


def pick_posts(n, seed):
    b = pd.read_csv(cfg["data"]["processed_dir"] + "/set_b.csv")
    L = b["text"].str.len()
    mid = b[(L > L.quantile(0.25)) & (L < L.quantile(0.75))]  # typical-length posts
    return mid.sample(n=n, random_state=seed)["text"].tolist()


def ollama_ps():
    models = requests.get(URL + "/api/ps", timeout=10).json()["models"]
    return [{"name": m["name"], "size_gb": round(m["size"] / 1e9, 2),
             "gpu_pct": round(100 * m["size_vram"] / m["size"]) if m["size"] else 0,
             "context": m.get("context_length")} for m in models]


def hardware():
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return {"gpu": gpu, "cpu": platform.processor(), "os": platform.platform(),
            "ollama": requests.get(URL + "/api/version").json()["version"]}


def run_batch(llm, posts, workers, model=None):
    """Run every call type on every post; poll `ollama ps` meanwhile. Returns a summary."""
    jobs = [(t, p) for p in posts for t in CALL_TYPES]

    def one(job):
        t, post = job
        system, schema = CALL_TYPES[t]
        r = llm.chat(system, "Post:\n" + post, schema=schema, model=model)
        return {"type": t, "tokens_in": r.tokens_in, "tokens_out": r.tokens_out,
                "latency": r.latency, "valid": r.valid, "attempts": r.attempts}

    ps_samples, done = [], threading.Event()

    def poll():
        while not done.is_set():
            ps_samples.append(ollama_ps())
            time.sleep(3)

    poller = threading.Thread(target=poll)
    poller.start()
    start = time.time()
    records = llm.map(one, jobs, workers=workers)
    wall = time.time() - start
    done.set()
    poller.join()

    df = pd.DataFrame(records)
    by_type = df.groupby("type").agg(calls=("latency", "size"), sec_per_call=("latency", "mean"),
                                     tokens_in=("tokens_in", "mean"), tokens_out=("tokens_out", "mean"),
                                     valid_rate=("valid", "mean"), repair_retries=("attempts", lambda a: int((a > 1).sum())))
    return {"workers": workers, "wall_seconds": round(wall, 1), "calls": len(df),
            "wall_seconds_per_call": round(wall / len(df), 2),
            "output_tokens_per_wall_second": round(df["tokens_out"].sum() / wall, 1),
            "by_type": by_type.round(2).to_dict("index"),
            "min_gpu_pct": min((m["gpu_pct"] for s in ps_samples for m in s), default=None),
            "ps_last": ps_samples[-1] if ps_samples else None}


def warm(model):
    requests.post(URL + "/api/chat", json={"model": model, "stream": False, "think": False,
                                           "messages": [{"role": "user", "content": "Say ok."}],
                                           "options": {"num_ctx": cfg["ollama"]["num_ctx"], "num_predict": 5}})


def part_parallel():
    llm, posts = make_client(), pick_posts(8, cfg["seed"])
    warm(cfg["models"]["main"])
    out = {}
    for workers in [2, 1]:  # parallel first, so a warm prompt cache can only help serial
        print(f"{workers} request(s) at a time...")
        out[f"workers_{workers}"] = run_batch(llm, posts, workers)
        print(json.dumps(out[f"workers_{workers}"], indent=2))
    return out


def part_models():
    llm, posts = make_client(), pick_posts(8, cfg["seed"])
    out = {"thinking": {}}
    warm(cfg["models"]["main"])
    for think in [False, True]:
        lat, toks = [], []
        for post in posts[:2]:
            payload = {"model": cfg["models"]["main"], "stream": False, "think": think,
                       "messages": [{"role": "system", "content": FULL_SYS},
                                    {"role": "user", "content": "Post:\n" + post}],
                       "format": FullOutput.model_json_schema(),
                       "options": {"temperature": 0, "num_ctx": cfg["ollama"]["num_ctx"], "seed": cfg["seed"]}}
            t0 = time.time()
            r = requests.post(URL + "/api/chat", json=payload, timeout=900).json()
            lat.append(time.time() - t0)
            toks.append(r["eval_count"])
        out["thinking"][f"think_{think}"] = {"sec_per_call": round(sum(lat) / 2, 1), "tokens_out": round(sum(toks) / 2)}
    for key in ["small", "strong"]:
        warm(cfg["models"][key])
        out[key] = run_batch(llm, posts[:4], workers=1, model=cfg["models"][key])
        out[key]["model"] = cfg["models"][key]
        print(json.dumps(out[key], indent=2))
    return out


def part_swap():
    """Alternate main and small model on a tiny prompt; report the load time of each switch."""
    loads = []
    for i in range(3):
        for key in ["main", "small"]:
            t0 = time.time()
            r = requests.post(URL + "/api/chat", json={
                "model": cfg["models"][key], "stream": False, "think": False,
                "messages": [{"role": "user", "content": "Say ok."}],
                "options": {"num_ctx": cfg["ollama"]["num_ctx"], "num_predict": 5}}).json()
            loads.append({"model": cfg["models"][key], "load_seconds": round(r["load_duration"] / 1e9, 2),
                          "wall_seconds": round(time.time() - t0, 2), "loaded_after": ollama_ps()})
            print(loads[-1])
    return {"switches": loads}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["parallel", "models", "swap"], required=True)
    part = ap.parse_args().part
    out = {"hardware": hardware(), **{"parallel": part_parallel, "models": part_models, "swap": part_swap}[part]()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{part}.json").write_text(json.dumps(out, indent=2))
    print("Saved", OUT / f"{part}.json")


if __name__ == "__main__":
    main()

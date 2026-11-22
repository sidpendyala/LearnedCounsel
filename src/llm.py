"""Ollama client used by every agent and LLM baseline.

What it adds on top of a plain HTTP call:
  - structured output: a pydantic model's JSON schema goes in Ollama's `format` field,
    the reply is validated, and one repair retry is made if it is invalid
  - a disk cache keyed by (model, messages, schema, temperature, sample_id), so
    interrupted runs resume and repeated runs cost nothing
  - token and latency accounting (thread-safe)
  - a runtime guard (stop when the live model time exceeds a budget)
  - num_ctx enforcement (Ollama silently trims prompts that do not fit)
  - retries with backoff on connection errors, and optional parallel calls

FakeLLM (bottom of the file) runs the same logic with canned replies, for tests.
"""
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
from pydantic import BaseModel, ValidationError

log = logging.getLogger("llm")

CHARS_PER_TOKEN = 3.5  # conservative estimate (English is ~4 chars/token)


class ContextTooLong(Exception):
    pass


class RuntimeBudgetExceeded(Exception):
    pass


@dataclass
class LLMResult:
    parsed: dict | None     # validated JSON (None if still invalid after the repair retry)
    raw: str                # final raw text from the model
    tokens_in: int
    tokens_out: int
    latency: float          # seconds of model time when the call was first made
    attempts: int           # 1, or 2 if a repair retry was needed
    valid: bool
    cached: bool = False


class LLMClient:
    def __init__(self, url, model, num_ctx=8192, num_predict=1024, seed=0,
                 cache_dir="cache/llm", max_seconds=None, ctx_warn_frac=0.85,
                 max_http_retries=3, keep_alive="30m"):
        self.url = url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.num_predict = num_predict      # cap on output tokens per call
        self.seed = seed
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_seconds = max_seconds      # runtime guard (live model seconds), None = no limit
        self.ctx_warn_frac = ctx_warn_frac
        self.max_http_retries = max_http_retries
        self.keep_alive = keep_alive
        self._lock = threading.Lock()
        self.stats = dict(calls=0, cached_calls=0, tokens_in=0, tokens_out=0,
                          model_seconds=0.0, live_seconds=0.0, repair_retries=0,
                          invalid_after_retry=0, ctx_warnings=0)

    # ---------- public API ----------
    def chat(self, system, user, schema=None, temperature=0.0, sample_id=0, model=None):
        """One LLM call. `schema` is a pydantic model class (or None for free text)."""
        model = model or self.model
        if "cloud" in model:
            raise ValueError(f"Cloud models are not allowed: {model}")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        schema_json = schema.model_json_schema() if schema else None
        key = self._cache_key(model, messages, schema_json, temperature, sample_id)

        cached = self._cache_load(key)
        if cached is not None:
            result = LLMResult(**cached, cached=True)
            self._record(result)
            return result

        self._check_context(messages)
        self._check_budget()

        resp = self._call(model, messages, schema_json, temperature, sample_id)
        parsed, error = self._parse(resp["message"]["content"], schema)
        tokens_in, tokens_out = resp.get("prompt_eval_count", 0), resp.get("eval_count", 0)
        latency, attempts = resp["_latency"], 1
        self._check_truncation(resp)

        if error is not None:  # one repair retry: show the model its reply and the error
            repair = messages + [
                {"role": "assistant", "content": resp["message"]["content"]},
                {"role": "user", "content": f"Your reply was invalid: {error[:300]}\n"
                                            "Reply again with only valid JSON that matches the schema."}]
            resp = self._call(model, repair, schema_json, temperature, sample_id)
            parsed, error = self._parse(resp["message"]["content"], schema)
            tokens_in += resp.get("prompt_eval_count", 0)
            tokens_out += resp.get("eval_count", 0)
            latency += resp["_latency"]
            attempts = 2

        result = LLMResult(parsed=parsed, raw=resp["message"]["content"], tokens_in=tokens_in,
                           tokens_out=tokens_out, latency=latency, attempts=attempts,
                           valid=error is None)
        self._cache_save(key, result)
        self._record(result)
        return result

    def map(self, fn, items, workers=1):
        """Apply fn to items, with `workers` requests in flight at once. Keeps order."""
        if workers <= 1:
            return [fn(x) for x in items]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, items))

    # ---------- internals ----------
    def _call(self, model, messages, schema_json, temperature, sample_id):
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx,
                        "num_predict": self.num_predict, "seed": self.seed + sample_id},
        }
        if schema_json is not None:
            payload["format"] = schema_json
        start = time.time()
        resp = self._post(payload)
        resp["_latency"] = time.time() - start
        return resp

    def _post(self, payload):
        """HTTP call with retries and exponential backoff (2s, 4s, 8s ...)."""
        for attempt in range(self.max_http_retries + 1):
            try:
                r = requests.post(self.url + "/api/chat", json=payload, timeout=900)
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                if attempt == self.max_http_retries:
                    raise
                wait = 2 ** (attempt + 1)
                log.warning("Ollama request failed (%s); retrying in %ss", e, wait)
                time.sleep(wait)

    @staticmethod
    def _parse(text, schema):
        """Return (parsed_dict, None) on success or (None, error_message)."""
        if schema is None:
            return {"text": text}, None
        try:
            return schema.model_validate_json(text).model_dump(), None
        except ValidationError as e:
            return None, str(e)

    def _check_context(self, messages):
        est = sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN
        budget = self.num_ctx - self.num_predict
        if est > budget:
            raise ContextTooLong(f"Prompt is ~{est:.0f} tokens; limit is {budget} "
                                 f"(num_ctx {self.num_ctx} - num_predict {self.num_predict})")
        if est > self.ctx_warn_frac * budget:
            log.warning("Prompt is ~%.0f tokens, close to the %d-token limit", est, budget)
            with self._lock:
                self.stats["ctx_warnings"] += 1

    def _check_truncation(self, resp):
        """After the call: Ollama's own token count tells us if we came close to num_ctx."""
        used = resp.get("prompt_eval_count", 0) + resp.get("eval_count", 0)
        if used >= self.ctx_warn_frac * self.num_ctx:
            log.warning("Call used %d of %d context tokens (possible truncation)", used, self.num_ctx)
            with self._lock:
                self.stats["ctx_warnings"] += 1

    def _check_budget(self):
        if self.max_seconds is not None and self.stats["live_seconds"] >= self.max_seconds:
            raise RuntimeBudgetExceeded(
                f"Live model time {self.stats['live_seconds']:.0f}s reached the budget of "
                f"{self.max_seconds:.0f}s. Re-run to resume from the cache.")

    def _record(self, result):
        with self._lock:
            s = self.stats
            s["calls"] += 1
            s["tokens_in"] += result.tokens_in
            s["tokens_out"] += result.tokens_out
            s["model_seconds"] += result.latency
            if result.cached:
                s["cached_calls"] += 1
            else:
                s["live_seconds"] += result.latency
                s["repair_retries"] += result.attempts - 1
                s["invalid_after_retry"] += 0 if result.valid else 1

    def _cache_key(self, model, messages, schema_json, temperature, sample_id):
        blob = json.dumps([model, messages, schema_json, temperature, sample_id, self.num_ctx],
                          sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key):
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cache_load(self, key):
        if self.cache_dir is None:
            return None
        path = self._cache_path(key)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _cache_save(self, key, result):
        if self.cache_dir is None:
            return
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(result)
        data.pop("cached")
        tmp = path.with_suffix(".tmp")  # write-then-rename, so a crash never leaves half a file
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, path)


def client_from_config(cfg, model=None, max_seconds=None, cache_dir="cache/llm"):
    return LLMClient(url=cfg["ollama"]["url"], model=model or cfg["models"]["main"],
                     num_ctx=cfg["ollama"]["num_ctx"], num_predict=cfg["ollama"]["num_predict"],
                     seed=cfg["seed"], cache_dir=cache_dir, max_seconds=max_seconds)


class FakeLLM(LLMClient):
    """LLMClient whose replies come from `reply_fn(payload) -> str` instead of Ollama.

    Everything else (cache, validation, repair retry, accounting, guards) is the real code.
    Token counts are faked as characters // 4.
    """

    def __init__(self, reply_fn, cache_dir=None, **kwargs):
        super().__init__(url="http://fake", model="fake-model", cache_dir=cache_dir, **kwargs)
        self.reply_fn = reply_fn
        self.payloads = []

    def _post(self, payload):
        self.payloads.append(payload)
        content = self.reply_fn(payload)
        prompt_chars = sum(len(m["content"]) for m in payload["messages"])
        return {"message": {"content": content},
                "prompt_eval_count": prompt_chars // 4, "eval_count": len(content) // 4}

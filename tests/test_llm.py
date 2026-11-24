"""Tests for src/llm.py. They use FakeLLM, so Ollama does not need to be running."""
import json
import time

import pytest
from pydantic import BaseModel

from src.llm import ContextTooLong, FakeLLM, RuntimeBudgetExceeded


class Answer(BaseModel):
    category: str
    present: bool


GOOD = json.dumps({"category": "housing", "present": True})


def test_valid_json_is_parsed_and_counted():
    llm = FakeLLM(lambda p: GOOD)
    r = llm.chat("sys", "user", schema=Answer)
    assert r.valid and r.parsed == {"category": "housing", "present": True}
    assert r.attempts == 1 and not r.cached
    assert llm.stats["calls"] == 1 and llm.stats["tokens_out"] > 0


def test_request_settings_follow_the_spec():
    llm = FakeLLM(lambda p: GOOD, num_ctx=8192)
    llm.chat("sys", "user", schema=Answer)
    p = llm.payloads[0]
    assert p["think"] is False
    assert p["options"]["num_ctx"] == 8192 and p["options"]["temperature"] == 0.0
    assert p["format"]["properties"].keys() == {"category", "present"}


def test_invalid_json_gets_one_repair_retry():
    replies = iter(["not json at all", GOOD])
    llm = FakeLLM(lambda p: next(replies))
    r = llm.chat("sys", "user", schema=Answer)
    assert r.valid and r.attempts == 2
    assert "invalid" in llm.payloads[1]["messages"][-1]["content"]
    assert llm.stats["repair_retries"] == 1


def test_still_invalid_after_retry_is_reported_not_raised():
    llm = FakeLLM(lambda p: '{"category": "housing"}')  # missing "present"
    r = llm.chat("sys", "user", schema=Answer)
    assert not r.valid and r.parsed is None and r.attempts == 2
    assert llm.stats["invalid_after_retry"] == 1


def test_cache_hit_skips_the_model(tmp_path):
    llm = FakeLLM(lambda p: GOOD, cache_dir=tmp_path)
    first = llm.chat("sys", "user", schema=Answer)
    second = llm.chat("sys", "user", schema=Answer)
    assert len(llm.payloads) == 1 and second.cached and second.parsed == first.parsed
    # a new client (e.g. after a crash) also resumes from the cache
    llm2 = FakeLLM(lambda p: pytest.fail("should not call the model"), cache_dir=tmp_path)
    assert llm2.chat("sys", "user", schema=Answer).cached


def test_cache_key_includes_temperature_and_sample_id(tmp_path):
    llm = FakeLLM(lambda p: GOOD, cache_dir=tmp_path)
    llm.chat("sys", "user", schema=Answer, temperature=0.7, sample_id=0)
    llm.chat("sys", "user", schema=Answer, temperature=0.7, sample_id=1)
    llm.chat("sys", "user", schema=Answer, temperature=0.0, sample_id=0)
    assert len(llm.payloads) == 3
    assert llm.payloads[0]["options"]["seed"] != llm.payloads[1]["options"]["seed"]


def test_prompt_too_long_raises():
    llm = FakeLLM(lambda p: GOOD, num_ctx=1000, num_predict=200)
    with pytest.raises(ContextTooLong):
        llm.chat("sys", "x" * 4000, schema=Answer)  # ~1140 tokens > 800


def test_prompt_close_to_limit_warns():
    llm = FakeLLM(lambda p: GOOD, num_ctx=1000, num_predict=200)
    llm.chat("sys", "x" * 2700, schema=Answer)  # ~770 tokens, above 85% of 800
    assert llm.stats["ctx_warnings"] >= 1


def test_runtime_guard_stops_new_calls_but_cache_still_works(tmp_path):
    llm = FakeLLM(lambda p: GOOD, cache_dir=tmp_path, max_seconds=10)
    llm.chat("sys", "a", schema=Answer)
    llm._wall = 11  # pretend we used up the budget
    assert llm.chat("sys", "a", schema=Answer).cached  # cached call is still fine
    with pytest.raises(RuntimeBudgetExceeded):
        llm.chat("sys", "b", schema=Answer)


def test_cloud_models_are_refused():
    llm = FakeLLM(lambda p: GOOD)
    with pytest.raises(ValueError):
        llm.chat("sys", "user", schema=Answer, model="qwen3:8b-cloud")


def test_parallel_map_keeps_order():
    llm = FakeLLM(lambda p: json.dumps({"category": p["messages"][1]["content"], "present": True}))
    out = llm.map(lambda c: llm.chat("sys", c, schema=Answer).parsed["category"],
                  [f"c{i}" for i in range(8)], workers=2)
    assert out == [f"c{i}" for i in range(8)]
    assert llm.stats["calls"] == 8


def test_wall_time_does_not_double_count_parallel_calls():
    def slow(p):
        time.sleep(0.2)
        return GOOD
    llm = FakeLLM(slow)
    llm.map(lambda i: llm.chat("sys", str(i), schema=Answer), range(4), workers=2)
    assert llm.stats["live_seconds"] > 0.7      # 4 calls x 0.2 s of latency
    assert 0.3 < llm.wall_seconds() < 0.7       # but only about 0.4 s of wall time

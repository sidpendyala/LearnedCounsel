"""v2 flow with a FakeLLM: scout, disagreements, misses-only skeptic, stricter note."""
import json

import pytest

from src import orchestrator
from src.llm import FakeLLM
from src.orchestrator import Settings, run_case
from src.schemas import CATEGORIES
from tests.test_orchestrator import POST, reply

V2 = dict(use_router=False, skeptic_misses_only=True, safe_note=True)


@pytest.fixture(autouse=True)
def tmp_transcripts(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "TRANSCRIPT_DIR", tmp_path)


def v2_reply(payload):
    title = payload["format"]["title"]
    if title == "B3Output":      # the scout sees housing and torts, but not consumer
        return json.dumps({c: {"present": c in ("housing", "torts"),
                               "evidence_quote": "the ceiling fell on my arm" if c == "torts" else ""} for c in CATEGORIES})
    if title == "SkepticMissesOutput":
        return json.dumps({"possible_misses": [{"category": "housing", "reason": "already present"},
                                               {"category": "health", "reason": "hurt arm"}]})
    return reply(payload)


def kinds(case):
    return [(d["kind"], d["category"]) for d in case.debate if d["type"] == "rebuttal"]


def test_scout_add_only_raises_issues_the_specialists_did_not_claim():
    case = run_case(FakeLLM(v2_reply), "p1", POST, Settings(scout_mode="add", **V2))
    assert len(case.specialist_outputs) == 5                       # no router
    assert ("scout_add", "torts") in kinds(case)
    assert not any(k == "scout_drop" for k, _ in kinds(case))
    assert ("possible_miss", "health") in kinds(case)              # misses on present categories are dropped
    assert not any(k == "challenge" for k, _ in kinds(case))


def test_scout_both_also_questions_claims_it_did_not_see():
    case = run_case(FakeLLM(v2_reply), "p1", POST, Settings(scout_mode="both", **V2))
    assert ("scout_drop", "consumer") in kinds(case)


def test_specialists_and_scout_judge_independently():
    llm = FakeLLM(v2_reply)
    run_case(llm, "p1", POST, Settings(scout_mode="add", **V2))
    by_title = {}
    for p in llm.payloads:
        by_title.setdefault(p["format"]["title"], p)
    assert "deposit dispute" not in json.dumps(by_title["B3Output"]["messages"])
    specialists = [p for p in llm.payloads if p["format"]["title"].startswith("Specialist_")]
    assert all("Scout" not in json.dumps(p["messages"]) for p in specialists)


def test_v2_is_off_by_default():
    llm = FakeLLM(v2_reply)
    case = run_case(llm, "p1", POST, Settings())
    assert case.scout is None and "B3Output" not in {p["format"]["title"] for p in llm.payloads}

"""End-to-end test of the agent flow with a FakeLLM (no Ollama needed).

The fake replies depend on which schema was requested (payload["format"]["title"]).
Story: a tenant post. Housing is found with high confidence; Work & Money claims
"consumer" with LOW confidence; the skeptic challenges consumer (allowed, below tau)
and employment (not allowed: not a positive claim) and flags torts as a possible miss.
"""
import json

import pytest

from src import orchestrator
from src.llm import FakeLLM
from src.orchestrator import Settings, learnedcounsel_post, run_case
from src.schemas import CATEGORIES, GROUPS

POST = "My landlord kept my security deposit and the ceiling fell on my arm."


def claim(present, conf, quote="", why="ok"):
    return {"present": present, "confidence": conf, "evidence_quote": quote, "rationale": why}


def reply(payload, coordinator_settled=True):
    title = payload["format"]["title"]
    user = payload["messages"][1]["content"]
    if title == "Intake":
        return json.dumps({"facts": ["deposit kept"], "parties": ["landlord"], "location_state": None,
                           "timeline": [], "user_goals": ["get deposit back"], "urgency": "medium",
                           "urgency_reasons": ["money owed"]})
    if title == "RouterOutput":
        return json.dumps({g: {"call": g in ("housing", "work_money"), "reason": "r"} for g in GROUPS})
    if title.startswith("Specialist_"):
        group = title.removeprefix("Specialist_")
        cats = GROUPS.get(group, CATEGORIES)
        out = {c: claim(False, 0.9) for c in cats}
        if "housing" in cats:
            out["housing"] = claim(True, 0.95, "My landlord kept my security deposit", "deposit dispute")
        if "consumer" in cats:
            out["consumer"] = claim(True, 0.6, "security deposit", "maybe a consumer debt")
        out["consult_requests"] = []
        return json.dumps(out)
    if title == "SkepticOutput":
        return json.dumps({"challenges": [
            {"category": "consumer", "type": "wrong_category", "argument": "a deposit is housing"},
            {"category": "employment", "type": "overclaim", "argument": "not allowed"}],
            "possible_misses": [{"category": "torts", "reason": "ceiling fell on arm"}]})
    if title == "Rebuttal":
        if '"consumer"' in payload["messages"][0]["content"]:
            return json.dumps({"decision": "withdraw", "present": False, "new_confidence": 0.8,
                               "evidence_quote": "", "response": "agree, it is housing"})
        return json.dumps({"decision": "revise", "present": True, "new_confidence": 0.8,
                           "evidence_quote": "the ceiling fell on my arm", "response": "injury"})
    if title == "CoordinatorOutput":
        rounds_left = int(user.rsplit("Rounds of questions left: ", 1)[1][0])
        labels = {c: {"present": c in ("housing", "torts"), "confidence": 0.9} for c in CATEGORIES}
        settled = coordinator_settled or rounds_left == 0
        return json.dumps({"labels": labels, "urgency": "medium", "referral_note": "Contact a tenant group.",
                           "settled": settled,
                           "reopen": [] if settled else [{"category": "torts", "question": "is it serious?"}]})
    raise AssertionError(f"unexpected schema {title}")


@pytest.fixture(autouse=True)
def tmp_transcripts(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "TRANSCRIPT_DIR", tmp_path)


def test_full_flow():
    llm = FakeLLM(reply)
    case = run_case(llm, "p1", POST, Settings())
    agents = [m["agent"] for m in case.messages]
    assert agents[:4] == ["intake", "router", "specialist:housing", "specialist:work_money"]
    # tau enforced: only the low-confidence positive claim (consumer) may be challenged
    skeptic = [d for d in case.debate if d["type"] == "skeptic"][0]
    assert [i["category"] for i in skeptic["items"]] == ["consumer", "torts"] and skeptic["dropped"] == 1
    flips = {d["category"]: (d["before"], d["after"]) for d in case.debate if d["type"] == "rebuttal"}
    assert flips == {"consumer": (True, False), "torts": (False, True)}
    assert case.final["labels"]["housing"] and case.final["labels"]["torts"]
    assert not case.final["labels"]["consumer"]


def test_specialists_do_not_see_each_other():
    llm = FakeLLM(reply)
    run_case(llm, "p1", POST, Settings())
    work_money_prompt = [p for p in llm.payloads if p["format"]["title"] == "Specialist_work_money"][0]
    text = json.dumps(work_money_prompt["messages"])
    assert "deposit dispute" not in text  # the housing specialist's rationale


def test_high_tau_lets_skeptic_challenge_more_and_zero_rounds_skip_debate():
    llm = FakeLLM(reply)
    case = run_case(llm, "p1", POST, Settings(max_rounds=0))
    assert not any(m["agent"] in ("skeptic",) or m["agent"].startswith("rebuttal") for m in case.messages)


def test_reopen_loop_is_capped_by_max_rounds():
    llm = FakeLLM(lambda p: reply(p, coordinator_settled=False))
    case = run_case(llm, "p1", POST, Settings(max_rounds=2))
    coordinators = [m for m in case.messages if m["agent"] == "coordinator"]
    assert len(coordinators) == 2 and case.final["rounds"] == 2  # round 1 skeptic, round 2 one reopen


def test_ablation_switches():
    llm = FakeLLM(reply)
    case = run_case(llm, "p1", POST, Settings(use_router=False, use_skeptic=False))
    assert len(case.specialist_outputs) == 5 and "skeptic" not in [m["agent"] for m in case.messages]
    case = run_case(FakeLLM(reply), "p1", POST, Settings(single_specialist=True))
    assert list(case.specialist_outputs) == ["general"]


def test_adapter_rows_and_costs():
    rows, cost = learnedcounsel_post(FakeLLM(reply), "p1", POST, Settings())
    assert len(rows) == 16 and cost["flips"] == 2 and cost["challenges"] == 1 and cost["calls"] >= 7

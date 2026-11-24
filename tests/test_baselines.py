"""LLM baselines with a FakeLLM (no Ollama needed)."""
import json

from src.baselines import b2_post, b3_post, b4_post, b5_post
from src.llm import FakeLLM
from src.schemas import CATEGORIES

POST = "My landlord kept my security deposit."


def b3_reply(present):
    return json.dumps({c: {"present": c in present, "evidence_quote": "landlord kept my security deposit"}
                       for c in CATEGORIES})


def positives(rows):
    return {r["category"] for r in rows if r["pred"]}


def test_b2_asks_only_the_set_a_categories_and_shows_only_that_definition():
    llm = FakeLLM(lambda p: json.dumps({"answer": "Yes" if '"housing"' in p["messages"][0]["content"] else "No"}))
    rows, cost = b2_post(llm, "p1", POST, {"p1": ["housing", "crime"]})
    assert [(r["category"], r["pred"]) for r in rows] == [("housing", 1), ("crime", 0)]
    assert cost["calls"] == 2
    assert "**crime**" not in llm.payloads[0]["messages"][0]["content"]


def test_b3_rows_and_invalid_output_counts_as_all_absent():
    rows, cost = b3_post(FakeLLM(lambda p: b3_reply({"housing"})), "p1", POST, "present_first")
    assert len(rows) == 16 and positives(rows) == {"housing"} and cost["valid"]
    assert [r["evidence_valid"] for r in rows if r["pred"]] == [True]
    rows, cost = b3_post(FakeLLM(lambda p: "not json"), "p1", POST, "present_first")
    assert positives(rows) == set() and not cost["valid"] and cost["calls"] == 2


def test_b4_uses_final_labels_not_the_first_pass():
    claim = {"evidence_quote": "", "rationale": "", "present": False, "confidence": 0.9}
    out = {"facts": [], "urgency": "low", "first_pass": {c: claim for c in CATEGORIES},
           "challenges": [], "final_labels": {c: c == "housing" for c in CATEGORIES}, "referral_note": ""}
    rows, _ = b4_post(FakeLLM(lambda p: json.dumps(out)), "p1", POST, tau=0.7)
    assert positives(rows) == {"housing"}


def test_b5_majority_vote_uses_different_samples():
    replies = iter([b3_reply({"housing", "crime"}), b3_reply({"housing"}), b3_reply({"housing", "torts"})])
    llm = FakeLLM(lambda p: next(replies))
    rows, cost = b5_post(llm, "p1", POST, "present_first", k=3, temperature=0.7)
    assert positives(rows) == {"housing"} and cost["calls"] == 3
    assert len({p["options"]["seed"] for p in llm.payloads}) == 3


def test_b5_ignores_invalid_samples():
    replies = iter([b3_reply({"housing"}), "bad", "bad", b3_reply({"housing"})])  # sample 2 fails twice
    rows, cost = b5_post(FakeLLM(lambda p: next(replies)), "p1", POST, "present_first", k=3, temperature=0.7)
    assert positives(rows) == {"housing"} and cost["invalid_messages"] == 1

import pandas as pd
import pytest

from src.debate import debate_metrics
from src.evaluate import score_set_a, score_set_b
from src.schemas import CATEGORIES
from src.stats import (holm, mcnemar, paired_difference, set_a_counts, set_a_metrics, set_b_counts,
                       set_b_metrics, with_ci)

CATS = ["housing", "family", "crime"]


def toy():
    set_b = pd.DataFrame({"post_id": ["p1", "p2", "p3"], "n_known_pos": [2, 1, 1],
                          "housing": [1, 0, 1], "family": [1, None, 0], "crime": [None, 1, 0]})
    set_b[CATS] = set_b[CATS].astype("Int8")
    preds = pd.DataFrame({"post_id": ["p1"] * 3 + ["p2"] * 3 + ["p3"] * 3, "category": CATS * 3,
                          "pred": [1, 0, 1, 1, 1, 1, 1, 0, 0]})
    return set_b, preds


def test_point_estimates_match_evaluate():
    set_b, preds = toy()
    ci = with_ci(set_b_counts, set_b_metrics, preds, set_b, CATS, n_boot=200)
    plain = score_set_b(preds, set_b, CATS)
    for k in ["micro_f1", "macro_f1", "micro_precision", "micro_recall", "exact_match"]:
        assert ci[k]["value"] == pytest.approx(plain[k])
        assert ci[k]["ci95"][0] <= ci[k]["value"] <= ci[k]["ci95"][1]
    set_a = pd.DataFrame({"item_id": list("abc"), "post_id": ["p1", "p2", "p3"],
                          "category": ["housing", "housing", "crime"], "label": [1, 0, 0]})
    ci = with_ci(set_a_counts, set_a_metrics, preds, set_a, CATS, n_boot=200)
    assert ci["mean_accuracy"]["value"] == pytest.approx(score_set_a(preds, set_a)["mean_accuracy"])


def test_identical_systems_never_differ():
    set_b, preds = toy()
    d = paired_difference(set_b_counts, set_b_metrics, preds, preds, set_b, CATS, n_boot=200)
    assert d["micro_f1"] == {"value": 0.0, "ci95": [0.0, 0.0], "significant": False}


def test_mcnemar_exact():
    set_a = pd.DataFrame({"post_id": [f"p{i}" for i in range(6)], "category": "housing", "label": 1})
    right, wrong = set_a.assign(pred=1), set_a.assign(pred=0)
    r = mcnemar(right, wrong, set_a)
    assert r["only_1_correct"] == 6 and r["only_2_correct"] == 0
    assert r["p_value"] == pytest.approx(2 * 0.5 ** 6)
    assert mcnemar(right, right, set_a)["p_value"] == 1.0


def test_holm():
    h = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h["a"] == pytest.approx(0.03)
    assert h["c"] == pytest.approx(0.06) and h["b"] == pytest.approx(0.06)


def test_debate_metrics_score_flips_against_known_labels():
    labels = {c: False for c in CATEGORIES}
    claims = {c: {"present": False} for c in CATEGORIES}
    claims["housing"] = {"present": True}
    case = {"post_id": "p1", "consults": [{"specialist": "courts_crime", "reason": ""}],
            "specialist_outputs": {
                "housing": {"housing": {"present": True, "confidence": 1.0}, "consult_requests": []},
                "work_money": {"consumer": {"present": True, "confidence": 0.8}},
                "courts_crime": {"crime": {"present": False, "confidence": 1.0}}},
            "claims": claims, "final": {"labels": {**labels, "housing": True, "torts": True}},
            "debate": [
                {"type": "skeptic", "dropped": 2, "items": []},
                {"type": "rebuttal", "kind": "challenge", "category": "consumer", "before": True, "after": False},
                {"type": "rebuttal", "kind": "reopen", "category": "estates", "before": False, "after": True}]}
    m = debate_metrics([case], {("p1", "consumer"): 0, ("p1", "housing"): 1})
    assert m["challenge"]["flips_vs_known_labels"] == {"correct": 1, "incorrect": 0, "unknown": 0}
    assert m["reopen"]["flips_vs_known_labels"]["unknown"] == 1
    assert m["coordinator_overrides"]["count"] == 1        # torts: final True, claim False
    assert m["stages"]["round0_specialists"]["fp"] == 1 and m["stages"]["after_debate"]["fp"] == 0

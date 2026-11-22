import pandas as pd
import pytest

from src.evaluate import binary_metrics, quote_in_post, score_set_a, score_set_b

CATS = ["housing", "family", "crime"]


def test_quote_check_allows_case_punctuation_and_whitespace():
    post = "My landlord kept my  deposit. Now I'm\nbeing evicted!"
    assert quote_in_post("landlord kept my deposit", post)
    assert quote_in_post("Now I'm being evicted", post)
    assert not quote_in_post("my landlord stole my car", post)
    assert not quote_in_post("", post)
    assert quote_in_post("> 4. No person shall park", "reads: &gt; 4. No person shall park here")


def test_binary_metrics():
    m = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert m["accuracy"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert binary_metrics([0, 0], [0, 0])["precision"] == 0.0  # undefined -> 0


def test_set_b_ignores_masked_labels():
    set_b = pd.DataFrame({"post_id": ["p1", "p2"], "n_known_pos": [2, 1],
                          "housing": [1, 0], "family": [1, None], "crime": [None, 1]})
    set_b[CATS] = set_b[CATS].astype("Int8")
    # p1: predicts housing only (misses family); predicts crime too, but crime is masked for p1
    # p2: predicts crime correctly; family is masked for p2
    preds = pd.DataFrame({"post_id": ["p1"] * 3 + ["p2"] * 3, "category": CATS * 2,
                          "pred": [1, 0, 1, 0, 1, 1]})
    s = score_set_b(preds, set_b, CATS)
    assert s["n_known_entries"] == 4            # masked entries dropped
    assert s["micro_recall"] == pytest.approx(2 / 3)
    assert s["micro_precision"] == 1.0          # p1's crime=1 is masked, so not a false positive
    assert s["exact_match"] == 0.5              # p2 fully right, p1 not
    assert s["recall_by_known_positives"] == {"1": 1.0, "2": 0.5}


def test_set_a_joins_on_post_and_category():
    set_a = pd.DataFrame({"item_id": ["housing-1", "crime-2"], "post_id": ["p1", "p2"],
                          "category": ["housing", "crime"], "label": [1, 0]})
    preds = pd.DataFrame({"post_id": ["p1", "p1", "p2"], "category": ["housing", "crime", "crime"],
                          "pred": [1, 1, 1]})
    s = score_set_a(preds, set_a)
    assert s["n_items"] == 2 and s["overall"]["accuracy"] == 0.5

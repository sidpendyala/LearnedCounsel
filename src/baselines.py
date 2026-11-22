"""Baselines. Each LLM baseline maps one post to (prediction rows, cost row).

B1  TF-IDF + logistic regression, one classifier per category, 5-fold CV on the test data
B3  single LLM call returning all 16 labels + evidence
(B2, B4, B5 are added in M5.)
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from src.evaluate import quote_in_post
from src.prompts import load_prompt
from src.schemas import B3Output, CATEGORIES


# ---------------- B1 ----------------
def run_b1(test_long, seed, n_folds=5):
    """Out-of-fold predictions for every test item, plus per-category fold accuracies.

    Folds are grouped by post_id, so a post duplicated within a task never sits in
    both the training and the test fold.
    """
    rows, fold_acc = [], {}
    for cat in CATEGORIES:
        d = test_long[test_long["category"] == cat].reset_index(drop=True)
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        accs = []
        for fold, (tr, te) in enumerate(cv.split(d["text"], d["label"], groups=d["post_id"])):
            model = make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
                                  LogisticRegression(max_iter=1000))
            model.fit(d["text"][tr], d["label"][tr])
            pred = model.predict(d["text"][te])
            accs.append(float((pred == d["label"][te]).mean()))
            rows.append(pd.DataFrame({"post_id": d["post_id"][te], "category": cat,
                                      "pred": pred, "fold": fold}))
        fold_acc[cat] = accs
    preds = pd.concat(rows).drop_duplicates(["post_id", "category"]).reset_index(drop=True)
    return preds, fold_acc


# ---------------- B3 ----------------
B3_SYSTEM = load_prompt("b3_single")


def b3_post(llm, post_id, text):
    r = llm.chat(B3_SYSTEM, "Post:\n" + text, schema=B3Output)
    rows = []
    for cat in CATEGORIES:
        ans = r.parsed[cat] if r.valid else {"present": False, "evidence_quote": ""}
        present = bool(ans["present"])
        rows.append({"post_id": post_id, "category": cat, "pred": int(present),
                     "evidence_quote": ans["evidence_quote"] if present else "",
                     "evidence_valid": quote_in_post(ans["evidence_quote"], text) if present else None})
    cost = {"post_id": post_id, "calls": r.attempts, "tokens_in": r.tokens_in,
            "tokens_out": r.tokens_out, "latency_s": r.latency, "valid": r.valid}
    return rows, cost


LLM_SYSTEMS = {"b3": b3_post}

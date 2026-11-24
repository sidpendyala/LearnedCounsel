"""Baselines. Each LLM baseline maps one post to (prediction rows, cost row).

B1  TF-IDF + logistic regression, one classifier per category, 5-fold CV on the test data
B2  one Yes/No call per (post, category) item (the LegalBench question, zero-shot; Set A only)
B3  single LLM call returning all 16 labels + evidence
B4  single LLM call running LearnedCounsel's workflow inside one context
B5  self-consistency: majority vote over k samples of B3 at temperature 0.7
"""
from functools import partial

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from src.evaluate import quote_in_post
from src.prompts import category_definitions, load_prompt
from src.schemas import CATEGORIES, B2Output, B3EvidenceFirstOutput, B3Output, B4Output


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


# ---------------- helpers for the LLM baselines ----------------
def pred_row(post_id, text, cat, present, quote):
    """One prediction row in the common format."""
    return {"post_id": post_id, "category": cat, "pred": int(present),
            "evidence_quote": quote if present else "",
            "evidence_valid": quote_in_post(quote, text) if present else None}


def cost_row(post_id, results):
    return {"post_id": post_id, "calls": sum(r.attempts for r in results),
            "tokens_in": sum(r.tokens_in for r in results),
            "tokens_out": sum(r.tokens_out for r in results),
            "latency_s": sum(r.latency for r in results),
            "valid": all(r.valid for r in results),
            "invalid_messages": sum(not r.valid for r in results),
            "repair_retries": sum(r.attempts - 1 for r in results)}


# ---------------- B2 ----------------
def b2_post(llm, post_id, text, asked):
    """`asked` maps post_id -> the categories that Set A asks about for this post."""
    rows, results = [], []
    for cat in asked[post_id]:
        system = load_prompt("b2_binary", category=cat, definitions=category_definitions([cat]))
        r = llm.chat(system, "Post:\n" + text, schema=B2Output)
        rows.append({"post_id": post_id, "category": cat,
                     "pred": int(r.valid and r.parsed["answer"] == "Yes")})
        results.append(r)
    return rows, cost_row(post_id, results)


# ---------------- B3 ----------------
B3_VERSIONS = {"present_first": ("b3_single", B3Output),
               "evidence_first": ("b3_evidence_first", B3EvidenceFirstOutput)}


def b3_call(llm, text, version, temperature=0.0, sample_id=0):
    prompt, schema = B3_VERSIONS[version]
    return llm.chat(load_prompt(prompt), "Post:\n" + text, schema=schema,
                    temperature=temperature, sample_id=sample_id)


def b3_post(llm, post_id, text, version):
    r = b3_call(llm, text, version)
    absent = {"present": False, "evidence_quote": ""}
    answers = r.parsed if r.valid else {c: absent for c in CATEGORIES}
    rows = [pred_row(post_id, text, c, answers[c]["present"], answers[c]["evidence_quote"])
            for c in CATEGORIES]
    return rows, cost_row(post_id, [r])


# ---------------- B4 ----------------
def b4_post(llm, post_id, text, tau):
    r = llm.chat(load_prompt("b4_workflow", tau=tau), "Post:\n" + text, schema=B4Output)
    rows = []
    for c in CATEGORIES:
        present = r.valid and r.parsed["final_labels"][c]
        quote = r.parsed["first_pass"][c]["evidence_quote"] if r.valid else ""
        rows.append(pred_row(post_id, text, c, present, quote))
    return rows, cost_row(post_id, [r])


# ---------------- B5 ----------------
def b5_post(llm, post_id, text, version, k, temperature):
    """Majority vote over k samples (k is odd, so no ties). Invalid samples do not vote."""
    results = [b3_call(llm, text, version, temperature, sample_id=i) for i in range(k)]
    valid = [r.parsed for r in results if r.valid]
    rows = []
    for c in CATEGORIES:
        yes = [p[c] for p in valid if p[c]["present"]]
        present = len(yes) > len(valid) / 2
        rows.append(pred_row(post_id, text, c, present, yes[0]["evidence_quote"] if yes else ""))
    return rows, cost_row(post_id, results)


LLM_BASELINES = ["b2", "b3", "b3_present_first", "b3_evidence_first", "b4", "b5"]


def llm_baselines(cfg, asked):
    b = cfg["baselines"]
    return {"b2": partial(b2_post, asked=asked),
            "b3": partial(b3_post, version=b["b3_version"]),
            "b3_present_first": partial(b3_post, version="present_first"),
            "b3_evidence_first": partial(b3_post, version="evidence_first"),
            "b4": partial(b4_post, tau=cfg["learnedcounsel"]["tau"]),
            "b5": partial(b5_post, version=b["b3_version"], k=b["b5_k"],
                          temperature=b["b5_temperature"])}

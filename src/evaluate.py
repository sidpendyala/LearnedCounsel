"""Metrics. Every system writes predictions in one common format:
one row per (post_id, category) with pred in {0, 1}.

  Set A: binary metrics per category, joined on (post_id, category).
  Set B: masked multi-label metrics; entries whose label is unknown are ignored.
"""
import html
import re

import pandas as pd

from src.data import positives_bucket


def quote_in_post(quote, post):
    """True if `quote` appears in `post`, ignoring case, punctuation, extra whitespace
    and HTML entities (Reddit text contains e.g. '&gt;' where the model writes '>')."""
    def norm(s):
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", html.unescape(s).lower())).strip()
    q = norm(quote or "")
    return bool(q) and q in norm(post)


def binary_metrics(y_true, y_pred):
    """Accuracy, precision, recall, F1. Precision/recall are 0 when undefined."""
    y_true, y_pred = list(y_true), list(y_pred)
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    n = len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / n if n else 0.0
    return {"n": n, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def score_set_a(preds, set_a):
    """Binary metrics on Set A items whose post has a prediction."""
    m = set_a.merge(preds[["post_id", "category", "pred"]], on=["post_id", "category"], how="inner")
    per_cat = {c: binary_metrics(g["label"], g["pred"]) for c, g in m.groupby("category")}
    return {"n_items": len(m), "n_items_in_set": len(set_a),
            "mean_accuracy": sum(v["accuracy"] for v in per_cat.values()) / len(per_cat) if per_cat else None,
            "overall": binary_metrics(m["label"], m["pred"]),
            "per_category": per_cat}


def known_labels_long(set_b, categories):
    """Set B labels as rows (post_id, category, label), dropping masked entries."""
    long = set_b.melt(id_vars=["post_id", "n_known_pos"], value_vars=categories,
                      var_name="category", value_name="label")
    long = long.dropna(subset=["label"])
    long["label"] = long["label"].astype(int)
    return long


def score_set_b(preds, set_b, categories):
    """Masked multi-label metrics on Set B posts that have predictions."""
    m = known_labels_long(set_b, categories).merge(
        preds[["post_id", "category", "pred"]], on=["post_id", "category"], how="inner")
    micro = binary_metrics(m["label"], m["pred"])
    per_cat = {c: binary_metrics(g["label"], g["pred"]) for c, g in m.groupby("category")}
    with_pos = {c: v for c, v in per_cat.items() if v["tp"] + v["fn"] > 0}  # F1 needs a known positive
    m["correct"] = m["label"] == m["pred"]
    exact = m.groupby("post_id")["correct"].all()
    pos = m[m["label"] == 1]
    recall_by_n = {b: float(g["pred"].mean()) for b, g in pos.groupby(pos["n_known_pos"].map(positives_bucket))}
    return {"n_posts": int(m["post_id"].nunique()), "n_known_entries": len(m),
            "micro_precision": micro["precision"], "micro_recall": micro["recall"], "micro_f1": micro["f1"],
            "macro_f1": sum(v["f1"] for v in with_pos.values()) / len(with_pos) if with_pos else None,
            "macro_f1_categories": len(with_pos),
            "exact_match": float(exact.mean()) if len(exact) else None,
            "recall_by_known_positives": recall_by_n,
            "per_category": per_cat}


def cost_summary(costs):
    """Per-post averages from a table with columns calls, tokens_in, tokens_out, latency_s."""
    cols = ["calls", "tokens_in", "tokens_out", "latency_s"]
    out = {f"{c}_per_post": float(costs[c].mean()) for c in cols}
    out.update({f"total_{c}": float(costs[c].sum()) for c in cols})
    out["n_posts"] = len(costs)
    return out

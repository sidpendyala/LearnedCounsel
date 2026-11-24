"""Uncertainty: bootstrap confidence intervals, paired differences, McNemar's test.

Posts (not single labels) are resampled, because labels of the same post are related.
The same seed gives every system the same resamples, so differences are paired.
"""
import numpy as np
from scipy.stats import binomtest

from src.evaluate import known_labels_long


def resamples(n_posts, n_boot, seed):
    return np.random.default_rng(seed).integers(0, n_posts, size=(n_boot, n_posts))


def counts(merged, categories):
    """tp/fp/fn/correct/n per (post, category), posts in sorted order."""
    posts = sorted(merged["post_id"].unique())
    pi = {p: i for i, p in enumerate(posts)}
    ci = {c: i for i, c in enumerate(categories)}
    out = {k: np.zeros((len(posts), len(categories))) for k in ["tp", "fp", "fn", "correct", "n"]}
    for p, c, y, yhat in merged[["post_id", "category", "label", "pred"]].itertuples(index=False):
        i, j = pi[p], ci[c]
        out["tp"][i, j] += y == 1 and yhat == 1
        out["fp"][i, j] += y == 0 and yhat == 1
        out["fn"][i, j] += y == 1 and yhat == 0
        out["correct"][i, j] += y == yhat
        out["n"][i, j] += 1
    return posts, out


def set_b_counts(preds, set_b, categories):
    m = known_labels_long(set_b, categories).merge(preds[["post_id", "category", "pred"]],
                                                   on=["post_id", "category"])
    return counts(m, categories)


def set_a_counts(preds, set_a, categories):
    m = set_a.merge(preds[["post_id", "category", "pred"]], on=["post_id", "category"])
    return counts(m, categories)


def ratio(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b > 0)


def set_b_metrics(c, idx):
    """Metrics of each resample (one row of idx). c = counts from set_b_counts."""
    tp, fp, fn = (c[k][idx].sum(axis=1) for k in ["tp", "fp", "fn"])   # (n_boot, n_categories)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    has_pos = (tp + fn) > 0                                            # macro-F1 needs a known positive
    TP, FP, FN = tp.sum(1), fp.sum(1), fn.sum(1)
    all_right = (c["correct"] == c["n"]).all(axis=1).astype(float)
    return {"micro_f1": ratio(2 * TP, 2 * TP + FP + FN), "micro_precision": ratio(TP, TP + FP),
            "micro_recall": ratio(TP, TP + FN), "macro_f1": (f1 * has_pos).sum(1) / has_pos.sum(1),
            "exact_match": all_right[idx].mean(axis=1)}


def set_a_metrics(c, idx):
    correct, n = c["correct"][idx].sum(axis=1), c["n"][idx].sum(axis=1)
    return {"mean_accuracy": (ratio(correct, n) * (n > 0)).sum(1) / (n > 0).sum(1)}


def summarize(point, boot):
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"value": float(point), "ci95": [float(lo), float(hi)]}


def with_ci(count_fn, metric_fn, preds, data, categories, n_boot=1000, seed=0):
    posts, c = count_fn(preds, data, categories)
    if not posts:
        return None
    everyone = np.arange(len(posts))[None, :]
    point, boot = metric_fn(c, everyone), metric_fn(c, resamples(len(posts), n_boot, seed))
    return {k: summarize(point[k][0], boot[k]) for k in point}


def paired_difference(count_fn, metric_fn, preds_1, preds_2, data, categories, n_boot=1000, seed=0):
    """Metric of system 1 minus system 2, on the same resampled posts."""
    posts_1, c1 = count_fn(preds_1, data, categories)
    posts_2, c2 = count_fn(preds_2, data, categories)
    assert posts_1 == posts_2, "systems were run on different posts"
    idx, everyone = resamples(len(posts_1), n_boot, seed), np.arange(len(posts_1))[None, :]
    p1, p2 = metric_fn(c1, everyone), metric_fn(c2, everyone)
    b1, b2 = metric_fn(c1, idx), metric_fn(c2, idx)
    out = {k: summarize(p1[k][0] - p2[k][0], b1[k] - b2[k]) for k in p1}
    for v in out.values():
        v["significant"] = bool(v["ci95"][0] > 0 or v["ci95"][1] < 0)   # CI excludes 0
    return out


def mcnemar(preds_1, preds_2, set_a):
    """Exact McNemar test on Set A items answered by both systems."""
    key = ["post_id", "category"]
    m = set_a.merge(preds_1[key + ["pred"]], on=key)
    m = m.merge(preds_2[key + ["pred"]], on=key, suffixes=("_1", "_2"))
    ok1, ok2 = m["pred_1"] == m["label"], m["pred_2"] == m["label"]
    only_1, only_2 = int((ok1 & ~ok2).sum()), int((~ok1 & ok2).sum())
    p = binomtest(only_1, only_1 + only_2, 0.5).pvalue if only_1 + only_2 else 1.0
    return {"n_items": len(m), "only_1_correct": only_1, "only_2_correct": only_2, "p_value": float(p)}


def holm(p_values):
    """Holm correction for several comparisons. Input and output: {name: p}."""
    out, worst = {}, 0.0
    for i, (name, p) in enumerate(sorted(p_values.items(), key=lambda kv: kv[1])):
        worst = max(worst, min(1.0, p * (len(p_values) - i)))
        out[name] = worst
    return out

"""Data loading and evaluation-set construction.

Source: the 16 `learned_hands_*` tasks of LegalBench (Hugging Face "nguha/legalbench"),
read as plain TSV files at a pinned commit. Each task is a balanced Yes/No question
("does this post involve <category>?") with columns: index, text, answer.

Builds:
  Set A  - standard binary items, N_A per category (stratified Yes/No), fixed seed.
  Set B  - every unique post with a 16-dim label vector:
           1 = positive in that task, 0 = negative in that task,
           empty (NaN) = masked, because the post never appears in that task.

Run:  python -m src.data
"""
import json
import re
from pathlib import Path

import pandas as pd
import yaml
from huggingface_hub import hf_hub_download


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(text):
    """Lowercase and collapse whitespace. Used only to detect duplicate posts."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def make_post_id(text):
    """Stable short id for a post, derived from its normalized text."""
    import hashlib
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()[:12]


def load_split(cfg, split):
    """All 16 tasks for one split, as one long table: one row per (task, item)."""
    frames = []
    for cat in cfg["categories"]:
        path = hf_hub_download(cfg["data"]["hf_repo"], f"data/learned_hands_{cat}/{split}.tsv",
                               repo_type="dataset", revision=cfg["data"]["hf_revision"])
        df = pd.read_csv(path, sep="\t")
        assert list(df.columns) == ["index", "text", "answer"], df.columns
        assert set(df["answer"]) <= {"Yes", "No"}
        frames.append(pd.DataFrame({
            "item_id": [f"{cat}-{i}" for i in df["index"]],
            "category": cat,
            "text": df["text"],
            "label": (df["answer"] == "Yes").astype(int),
        }))
    long = pd.concat(frames, ignore_index=True)
    long.insert(1, "post_id", long["text"].map(make_post_id))
    return long


def build_set_a(test_long, n_per_cat, seed):
    """Stratified sample: ceil(n/2) positives and floor(n/2) negatives per category."""
    parts = []
    for cat, rows in test_long.groupby("category"):
        for label, k in [(1, (n_per_cat + 1) // 2), (0, n_per_cat // 2)]:
            pool = rows[rows["label"] == label]
            parts.append(pool.sample(n=min(k, len(pool)), random_state=seed))
    return pd.concat(parts).sort_values(["category", "item_id"]).reset_index(drop=True)


def build_set_b(test_long, categories):
    """One row per unique post. A (post, category) pair seen with conflicting labels is masked."""
    per_pair = test_long.groupby(["post_id", "category"])["label"].agg(["min", "max"])
    per_pair["value"] = per_pair["min"].where(per_pair["min"] == per_pair["max"])  # NaN on conflict
    labels = per_pair["value"].unstack("category").reindex(columns=categories).astype("Int8")
    texts = test_long.drop_duplicates("post_id").set_index("post_id")["text"]
    set_b = labels.join(texts).reset_index()
    set_b["n_known"] = set_b[categories].notna().sum(axis=1)
    set_b["n_known_pos"] = (set_b[categories] == 1).sum(axis=1)
    return set_b[["post_id", "text", "n_known", "n_known_pos"] + categories]


def positives_bucket(n):
    return "3+" if n >= 3 else str(n)


def stratified_sample(set_b, sizes, seed):
    """Sample posts by number of known positives, e.g. sizes = {"1": 40, "2": 50, "3+": 30}."""
    bucket = set_b["n_known_pos"].map(positives_bucket)
    parts = [set_b[bucket == b].sample(n=n, random_state=seed) for b, n in sizes.items()]
    return pd.concat(parts).reset_index(drop=True)


def overlap_stats(test_long, set_b, set_a, categories):
    pair_counts = test_long.groupby(["post_id", "category"]).size()
    conflicts = test_long.groupby(["post_id", "category"])["label"].nunique()
    pos = set_b[categories].eq(1).fillna(False).astype(int)  # masked counts as "not a known positive"
    co_pos = pos.T.dot(pos)
    pairs = [(a, b, int(co_pos.loc[a, b])) for i, a in enumerate(categories)
             for b in categories[i + 1:] if co_pos.loc[a, b] > 0]
    pairs.sort(key=lambda p: -p[2])
    chars = set_b["text"].str.len()
    return {
        "test_rows": len(test_long),
        "unique_posts": len(set_b),
        "duplicate_rows_within_same_task": int((pair_counts > 1).sum()),
        "conflicting_labels_within_same_task": int((conflicts > 1).sum()),
        "rows_per_category": test_long["category"].value_counts().sort_index().to_dict(),
        "tasks_per_post": {int(k): int(v) for k, v in set_b["n_known"].value_counts().sort_index().items()},
        "known_positives_per_post": {int(k): int(v) for k, v in set_b["n_known_pos"].value_counts().sort_index().items()},
        "posts_with_2plus_known_positives": int((set_b["n_known_pos"] >= 2).sum()),
        "top_co_positive_pairs": pairs[:15],
        "text_chars": {k: round(float(v)) for k, v in chars.describe().items()},
        "set_a_items": len(set_a),
        "set_a_unique_posts": int(set_a["post_id"].nunique()),
    }


def main():
    cfg = load_config()
    cats = cfg["categories"]
    out = Path(cfg["data"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)

    train = load_split(cfg, "train")
    test = load_split(cfg, "test")
    dev_ids = set(Path(cfg["data"]["dev_ids"]).read_text(encoding="utf-8").split())
    set_a = build_set_a(test[~test["post_id"].isin(dev_ids)], cfg["data"]["n_a_per_category"], cfg["seed"])
    set_b = build_set_b(test, cats)
    dev = set_b[set_b["post_id"].isin(dev_ids)]       # tuning only
    eval_pool = set_b[~set_b["post_id"].isin(dev_ids)]
    stats = overlap_stats(test, set_b, set_a, cats)
    stats["train_rows"] = len(train)

    # Evaluation samples sized to the compute budget (see NOTES.md, M2)
    set_b_sample = stratified_sample(eval_pool, cfg["data"]["set_b_sample"], cfg["seed"])
    ablation = stratified_sample(set_b_sample, cfg["data"]["ablation_subset"], cfg["seed"])
    all_posts = set(set_a["post_id"]) | set(set_b_sample["post_id"])
    assert not dev_ids & (all_posts | set(ablation["post_id"]))

    # v2: new dev and fresh test posts, disjoint from everything used so far (and from train posts)
    unused = eval_pool[~eval_pool["post_id"].isin(all_posts | set(train["post_id"]))]
    v2_test = stratified_sample(unused, cfg["data"]["v2_test"], cfg["seed"])
    unused = unused[~unused["post_id"].isin(v2_test["post_id"])]
    v2_dev = stratified_sample(unused, cfg["data"]["v2_dev"], cfg["seed"])
    groups = [dev_ids, all_posts, set(train["post_id"]) - all_posts - dev_ids, set(v2_test["post_id"]), set(v2_dev["post_id"])]
    assert len(set().union(*groups)) == sum(len(g) for g in groups)   # no post is in two groups
    stats["v2_dev_posts"], stats["v2_test_posts"] = len(v2_dev), len(v2_test)
    stats["dev_posts"] = len(dev)
    stats["train_posts_also_in_test"] = len(set(train["post_id"]) & set(test["post_id"]))
    stats["set_b_sample_posts"] = len(set_b_sample)
    stats["ablation_subset_posts"] = len(ablation)
    stats["unique_posts_to_run_main"] = len(all_posts)

    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test_long.csv", index=False)
    set_a.to_csv(out / "set_a.csv", index=False)
    set_b.to_csv(out / "set_b.csv", index=False)
    set_b_sample.to_csv(out / "set_b_sample.csv", index=False)
    ablation.to_csv(out / "ablation_subset.csv", index=False)
    dev.to_csv(out / "dev.csv", index=False)
    v2_dev.to_csv(out / "v2_dev.csv", index=False)
    v2_test.to_csv(out / "v2_test.csv", index=False)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    if stats["posts_with_2plus_known_positives"] < cfg["data"]["hand_label_threshold"]:
        print("Fewer multi-issue posts than the threshold: build data/hand_label_template.csv")


if __name__ == "__main__":
    main()

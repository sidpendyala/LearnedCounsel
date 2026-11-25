"""Few-shot examples for experiment (h): labeled posts from the LegalBench train split.

82 train posts also appear in the test split, so every evaluation and dev post is excluded.
For each category we show the shortest Yes post and the shortest No post, in full.
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.data import load_config
from src.prompts import load_prompt


@lru_cache
def usable_examples():
    d = Path(load_config()["data"]["processed_dir"])
    used = set()
    for name in ["set_a", "set_b_sample", "dev"]:
        used |= set(pd.read_csv(d / f"{name}.csv")["post_id"])
    train = pd.read_csv(d / "train.csv")
    return train[~train["post_id"].isin(used)]


def pick_examples(categories, pool=None):
    """[(category, label, text)]: the shortest Yes and the shortest No post of each category."""
    pool = usable_examples() if pool is None else pool
    out = []
    for c in categories:
        for label in [1, 0]:
            rows = pool[(pool["category"] == c) & (pool["label"] == label)]
            best = rows.loc[rows["text"].str.len().idxmin()]
            out.append((c, label, best["text"]))
    return out


@lru_cache
def few_shot_block(categories):
    """`categories` is a tuple. Returns the text added to a specialist's prompt."""
    lines = [f'Post: {text}\n{c}: {"present" if label else "absent"}'
             for c, label, text in pick_examples(categories)]
    return load_prompt("few_shot", examples="\n\n".join(lines))

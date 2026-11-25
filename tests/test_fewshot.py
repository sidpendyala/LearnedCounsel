import pandas as pd

from src.fewshot import pick_examples


def test_picks_the_shortest_yes_and_no_post_per_category():
    pool = pd.DataFrame({"category": ["housing"] * 4, "label": [1, 1, 0, 0],
                         "text": ["long yes post", "yes", "no", "long no post"]})
    assert pick_examples(["housing"], pool) == [("housing", 1, "yes"), ("housing", 0, "no")]

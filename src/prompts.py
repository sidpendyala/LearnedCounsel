"""Load prompt templates from prompts/*.md and fill in {placeholders}."""
import re
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(file, /, **values):
    """`file` is positional-only so a {name} placeholder can be passed as a keyword."""
    text = (PROMPT_DIR / f"{file}.md").read_text(encoding="utf-8")
    if "{definitions}" in text and "definitions" not in values:
        values["definitions"] = load_prompt("category_definitions")
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    return text.strip()


def category_definitions(categories=None):
    """The definitions file split per category; optionally only the given categories."""
    text = load_prompt("category_definitions")
    defs = dict(re.findall(r"\*\*(\w+)\*\*: (.+)", text))
    if categories is None:
        return defs
    return "\n".join(f"**{c}**: {defs[c]}" for c in categories)

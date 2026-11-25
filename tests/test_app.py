"""The demo page with a FakeLLM (no Ollama needed)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import src.llm
from src import orchestrator
from src.llm import FakeLLM
from tests.test_orchestrator import POST, reply

APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture
def app(monkeypatch):
    used = {}

    def fake_client(cfg, cache_dir="cache/llm", **kwargs):
        used["cache_dir"] = cache_dir
        return FakeLLM(reply)
    monkeypatch.setattr(src.llm, "client_from_config", fake_client)
    monkeypatch.setattr(orchestrator, "TRANSCRIPT_DIR", orchestrator.TRANSCRIPT_DIR)  # restored after the test
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.used = used
    return at


def text(at):
    return " ".join(str(e.value) for kind in ["markdown", "warning", "info", "caption", "title"] for e in at.get(kind))


def test_page_shows_disclaimer_and_names(app):
    assert not app.exception
    t = text(app)
    assert "not legal advice" in t and "Siddharth Pendyala" in t and "Palash Doshi" in t


def test_own_story_runs_and_is_not_cached(app):
    app.radio[0].set_value("Write my own").run()
    app.text_area[0].set_value(POST)
    app.button[0].click().run()
    assert not app.exception
    assert app.used["cache_dir"] is None           # a visitor's story is never written to disk
    t = text(app)
    assert "Contact a tenant group." in t and "housing" in t and "<mark>" in t
    assert "Your story was not saved" in t


def test_preset_uses_the_cache(app):
    app.button[0].click().run()
    assert not app.exception and app.used["cache_dir"] == "cache/llm"

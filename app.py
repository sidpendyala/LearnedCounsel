"""LearnedCounsel demo.   Run:  .venv/Scripts/python -m streamlit run app.py

The case file fills in live while the agents work. Everything runs locally through Ollama.
Preset stories use the disk cache, so the live demo is fast. Stories typed in by a visitor are
never written to disk: no cache, and the transcript goes to a temporary folder that is deleted.
"""
import html
import json
import re
import tempfile
import time
from pathlib import Path

import streamlit as st

from src import orchestrator
from src.data import load_config
from src.llm import client_from_config
from src.orchestrator import Settings, run_case
from src.schemas import CATEGORIES

DISCLAIMER = ("**This is not legal advice.** LearnedCounsel is a class project that sorts stories into legal "
              "issue categories. It makes mistakes, and its notes can be wrong. Talk to a lawyer or a legal aid "
              "organization about your situation. In an emergency, call 911.")
PRESETS = json.loads(Path("data/demo_presets.json").read_text(encoding="utf-8"))
OWN = "Write my own"
SPECIALIST = {"housing": "Housing", "family": "Family", "work_money": "Work & Money",
              "courts_crime": "Courts & Crime", "health_more": "Health & More"}


def nice(category):
    return category.replace("_", " ")


def highlight(story, quotes):
    """The story as HTML, with each evidence quote marked (ignoring case and extra spaces)."""
    out = html.escape(story)
    for q in sorted({q for q in quotes if q}, key=len, reverse=True):
        words = [re.escape(html.escape(w)) for w in q.strip(' "').split()]
        if words:
            out = re.sub(r"\s+".join(words), lambda m: f"<mark>{m.group(0)}</mark>", out, count=1, flags=re.I)
    return f'<div style="line-height:1.6;white-space:pre-wrap">{out}</div>'


def show_message(m):
    """One agent message, in plain words."""
    agent, o = m["agent"], m["output"]
    title = agent.split(":")[0].capitalize()
    if ":" in agent:
        title += f" ({SPECIALIST.get(agent.split(':')[1], agent.split(':')[1])})"
    if agent.endswith("evidence_retry"):
        title += ", asked again because a quote was not in the story"
    st.markdown(f"**{title}**" + (f" · round {m['round']}" if m["round"] else ""))
    if o is None:
        st.warning("This agent's answer was not valid JSON, so the team continued without it.")
    elif agent == "intake":
        st.write(f"Urgency: **{o['urgency']}** ({'; '.join(o['urgency_reasons'])})")
        st.write("Facts: " + "; ".join(o["facts"]))
    elif agent == "router":
        st.write("Calls: " + (", ".join(SPECIALIST[g] for g, v in o.items() if v["call"]) or "nobody"))
    elif agent.startswith("specialist"):
        found = {c: v for c, v in o.items() if c != "consult_requests" and v["present"]}
        for c, v in found.items():
            st.write(f"✅ **{nice(c)}** ({v['confidence']:.0%} sure): “{v['evidence_quote']}” — {v['rationale']}")
        if not found:
            st.write("Nothing in my area.")
        for q in o.get("consult_requests", []):
            st.write(f"🔎 Asks for {SPECIALIST[q['specialist']]}: {q['reason']}")
    elif agent == "skeptic":
        for c in o["challenges"]:
            st.write(f"❓ Challenges **{nice(c['category'])}** ({nice(c['type'])}): {c['argument']}")
        for c in o["possible_misses"]:
            st.write(f"👀 Possible miss **{nice(c['category'])}**: {c['reason']}")
        if not o["challenges"] and not o["possible_misses"]:
            st.write("No objections.")
        st.caption("A challenge is passed on only if the claim's confidence is below the threshold "
                   "or its quote is not in the story.")
    elif agent.startswith("rebuttal"):
        st.write(f"**{o['decision'].capitalize()}** → {'present' if o['present'] else 'absent'}: {o['response']}")
    elif agent == "coordinator":
        st.write("Labels: " + (", ".join(nice(c) for c, v in o["labels"].items() if v["present"]) or "none"))
        for q in o["reopen"] if not o["settled"] else []:
            st.write(f"↩️ Reopens **{nice(q['category'])}**: {q['question']}")
    st.divider()


def analyze(story, is_preset):
    cfg = load_config()
    llm = client_from_config(cfg, cache_dir="cache/llm" if is_preset else None)
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Case file")
        story_box = st.empty()
        story_box.markdown(highlight(story, []), unsafe_allow_html=True)
        st.markdown("**Issues so far**")
        claims_box = st.empty()
    with right:
        st.subheader("The team at work")
        status = st.status("Intake is reading the story...", expanded=True)

    def on_event(message, case):
        with status:
            show_message(message)
        present = {c: cl for c, cl in case.claims.items() if cl.present}
        claims_box.write(", ".join(f"**{nice(c)}** ({cl.confidence:.0%})" for c, cl in present.items()) or "none yet")
        story_box.markdown(highlight(story, [cl.evidence_quote for cl in present.values()]), unsafe_allow_html=True)
        status.update(label=f"{message['agent']} finished")

    lc = cfg["learnedcounsel"]
    settings = Settings(variant="demo", tau=lc["tau"], max_rounds=lc["max_rounds"], on_event=on_event)
    start = time.time()
    with tempfile.TemporaryDirectory() as tmp:   # the transcript of the demo is not kept
        orchestrator.TRANSCRIPT_DIR = Path(tmp)
        case = run_case(llm, "demo", story, settings)
    elapsed = time.time() - start
    status.update(label="Done", state="complete", expanded=True)

    labels = [c for c in CATEGORIES if case.final["labels"][c]]
    present = {c: case.claims[c] for c in labels}
    story_box.markdown(highlight(story, [cl.evidence_quote for cl in present.values() if cl.present]), unsafe_allow_html=True)
    claims_box.write(", ".join(f"**{nice(c)}**" for c in labels) or "none")

    st.header("Result")
    a, b = st.columns([1, 2])
    a.metric("Urgency", case.final["urgency"])
    b.markdown("**Legal issues:** " + (", ".join(nice(c) for c in labels) or "none found"))
    for c, cl in present.items():
        quote = f"“{cl.evidence_quote}”" if cl.present and cl.evidence_quote else "(no quote)"
        b.write(f"- **{nice(c)}**: {quote}" + ("" if cl.evidence_valid or not cl.present else " ⚠️ not found word for word in the story"))
    st.subheader("Referral note")
    st.info(case.final["referral_note"] or "(none)")
    st.warning(DISCLAIMER)

    changed = [d for d in case.debate if d["type"] == "rebuttal" and d["before"] != d["after"]]
    if changed:
        st.subheader("What the debate changed")
        for d in changed:
            who = {"challenge": "Skeptic's challenge", "possible_miss": "Skeptic's possible miss",
                   "reopen": "Coordinator's question"}[d["kind"]]
            st.write(f"- **{nice(d['category'])}**: {'present' if d['before'] else 'absent'} → "
                     f"{'present' if d['after'] else 'absent'} ({who}: {d['question']})")

    msgs = case.messages
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LLM calls", sum(m["attempts"] for m in msgs))
    c2.metric("Tokens in / out", f"{sum(m['tokens_in'] for m in msgs):,} / {sum(m['tokens_out'] for m in msgs):,}")
    c3.metric("Model time", f"{sum(m['latency_s'] for m in msgs):.0f} s")
    c4.metric("This run", f"{elapsed:.1f} s")
    cached = sum(m["cached"] for m in msgs)
    st.caption(f"Everything ran locally with {llm.model} through Ollama; nothing left this computer. "
               + (f"{cached} of {len(msgs)} answers came from the saved preset run, so 'model time' is the original time. "
                  if cached else "") + ("" if is_preset else "Your story was not saved."))


def about():
    st.markdown("""
### LearnedCounsel: Multi-Agent Legal Issue Spotting for Access to Justice
**Team leader:** Siddharth Pendyala  \n**Team member:** Palash Doshi  \nCMPSC 448, Penn State

LearnedCounsel reads a person's story, finds which of 16 legal issue categories it involves (with quotes as evidence),
estimates urgency, and writes a short referral note. A router calls specialist agents, a skeptic challenges them,
and a coordinator decides. All agents use a small open model that runs on one laptop.

**Data:** LegalBench (Guha et al., 2023) and Learned Hands (Stanford Legal Design Lab and Suffolk LIT Lab), CC BY-NC-SA 4.0.
The preset stories in this demo are made up and are not from the dataset.

**Name:** LearnedCounsel is named after the Learned Hands dataset, itself a nod to Judge Learned Hand.
It is unrelated to the commercial products Learned Hand AI or CoCounsel.
""")
    st.warning(DISCLAIMER)


def main():
    st.set_page_config(page_title="LearnedCounsel", page_icon="⚖️", layout="wide")
    st.title("LearnedCounsel")
    st.warning(DISCLAIMER)
    demo, info = st.tabs(["Demo", "About"])
    with info:
        about()
    with demo:
        choice = st.radio("Story", [p["title"] for p in PRESETS] + [OWN], horizontal=True)
        preset = next((p for p in PRESETS if p["title"] == choice), None)
        story = st.text_area("Tell the story in your own words", value=preset["story"] if preset else "",
                             height=180, key=f"story_{choice}")
        if st.button("Analyze", type="primary"):
            if not story.strip():
                st.error("Please write a story first.")
            else:
                analyze(story.strip(), is_preset=bool(preset) and story.strip() == preset["story"].strip())


main()

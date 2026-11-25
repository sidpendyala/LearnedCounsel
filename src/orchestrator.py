"""LearnedCounsel control flow: the agents work on one shared case file.

  intake -> router -> specialists (+ consults) -> [skeptic -> rebuttals]          (v1)
  intake -> all specialists + scout -> disagreements -> [skeptic -> rebuttals]    (v2)
         -> coordinator -> (if not settled and rounds left: reopen questions -> rebuttals -> coordinator)

Settings switch parts on and off for the ablations. Every LLM call is logged in
logs/transcripts/<variant>/<post_id>.jsonl.

Show a saved transcript:  python -m src.orchestrator logs/transcripts/full/<post_id>.jsonl
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src import agents
from src.schemas import CATEGORIES, CaseFile

TRANSCRIPT_DIR = Path("logs/transcripts")


@dataclass
class Settings:
    variant: str = "full"
    tau: float = 0.7                    # skeptic may challenge positive claims below this confidence
    max_rounds: int = 2                 # challenge/rebuttal rounds; 0 = no debate
    use_router: bool = True             # ablation (b)
    use_skeptic: bool = True            # ablation (a)
    use_consults: bool = True           # ablation (c)
    single_specialist: bool = False     # ablation (d)
    specialist_model: str | None = None # experiment (g); None = main model
    few_shot: bool = False              # experiment (h)
    scout_mode: str = "off"             # v2: off, "add" (scout may only add issues) or "both" (add or question)
    skeptic_misses_only: bool = False   # v2: the skeptic raises possible misses, no challenges
    safe_note: bool = False             # v2: stricter "no legal advice" rule for the referral note
    on_event: Callable | None = None    # callback after every LLM call (used by the demo)


def run_case(llm, post_id, text, s):
    case = CaseFile(post_id=post_id, post=text)
    agents.intake(llm, case, s)

    # Round 0: independent specialist judgments
    if s.single_specialist:
        agents.specialist(llm, case, s, "general")
    else:
        agents.router(llm, case, s)
        called = list(case.routing["specialists_to_call"])
        requests = []
        for g in called:
            requests += agents.specialist(llm, case, s, g)
        if s.use_consults:  # pull in specialists the router skipped (one level, no chains)
            for req in requests:
                g = req["specialist"]
                if g not in called:
                    called.append(g)
                    case.consults.append({"specialist": g, "reason": req["reason"]})
                    agents.specialist(llm, case, s, g)

    # v2: a Scout reads the post independently; the specialists answer where it disagrees with them
    if s.scout_mode != "off":
        agents.scout(llm, case, s)
        for item in agents.scout_items(case, s):
            agents.rebuttal(llm, case, s, item, 1)

    # Round 1: the skeptic challenges, the specialists answer
    round_ = 0
    if s.use_skeptic and s.max_rounds >= 1:
        round_ = 1
        for item in agents.skeptic(llm, case, s, round_):
            agents.rebuttal(llm, case, s, item, round_)

    # The coordinator decides, and may reopen questions while rounds are left
    decision = agents.coordinator(llm, case, s, round_, s.max_rounds - round_)
    while decision and not decision["settled"] and decision["reopen"] and round_ < s.max_rounds:
        round_ += 1
        for q in decision["reopen"]:
            agents.rebuttal(llm, case, s, {"category": q["category"], "kind": "reopen",
                                           "from": "Coordinator", "text": q["question"]}, round_)
        decision = agents.coordinator(llm, case, s, round_, s.max_rounds - round_)

    finalize(case, decision, round_)
    save_transcript(case, s.variant)
    return case


def finalize(case, decision, rounds):
    if decision is not None:
        labels = {c: decision["labels"][c]["present"] for c in CATEGORIES}
        conf = {c: decision["labels"][c]["confidence"] for c in CATEGORIES}
        urgency, note = decision["urgency"], decision["referral_note"]
    else:  # coordinator failed: fall back to the specialists' current claims
        labels = {c: case.claims[c].present for c in CATEGORIES}
        conf = {c: case.claims[c].confidence for c in CATEGORIES}
        urgency, note = (case.intake or {}).get("urgency", "medium"), ""
    case.final = {"labels": labels, "confidences": conf, "urgency": urgency, "referral_note": note,
                  "coordinator_failed": decision is None, "rounds": rounds}


def save_transcript(case, variant):
    path = TRANSCRIPT_DIR / variant / f"{case.post_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for m in case.messages:
            f.write(json.dumps(m) + "\n")
        f.write(json.dumps({"agent": "case_file", "case": case.model_dump(exclude={"messages"})}) + "\n")


def learnedcounsel_post(llm, post_id, text, settings):
    """Adapter for src.experiments: one post -> (prediction rows, cost row)."""
    case = run_case(llm, post_id, text, settings)
    rows = []
    for c in CATEGORIES:
        pred = bool(case.final["labels"][c])
        cl = case.claims[c]
        has_quote = pred and cl.present
        rows.append({"post_id": post_id, "category": c, "pred": int(pred),
                     "confidence": case.final["confidences"][c],
                     "evidence_quote": cl.evidence_quote if has_quote else "",
                     "evidence_valid": cl.evidence_valid if has_quote else None})
    rebuttals = [d for d in case.debate if d["type"] == "rebuttal"]
    msgs = case.messages
    cost = {"post_id": post_id, "calls": sum(m["attempts"] for m in msgs),
            "tokens_in": sum(m["tokens_in"] for m in msgs), "tokens_out": sum(m["tokens_out"] for m in msgs),
            "latency_s": sum(m["latency_s"] for m in msgs), "valid": all(m["valid"] for m in msgs),
            "invalid_messages": sum(not m["valid"] for m in msgs),
            "repair_retries": sum(m["attempts"] - 1 for m in msgs),
            "evidence_retries": sum(m["agent"].endswith("evidence_retry") for m in msgs),
            "first_try_evidence_valid": case.evidence["first_try_valid"],
            "first_try_evidence_total": case.evidence["first_try_total"],
            "specialists_called": len(case.specialist_outputs), "consults": len(case.consults),
            "challenges": sum(d["kind"] == "challenge" for d in rebuttals),
            "possible_misses": sum(d["kind"] == "possible_miss" for d in rebuttals),
            "reopens": sum(d["kind"] == "reopen" for d in rebuttals),
            "scout_questions": sum(d["kind"].startswith("scout") for d in rebuttals),
            "flips": sum(d["before"] != d["after"] for d in rebuttals),
            "rounds": case.final["rounds"], "coordinator_failed": case.final["coordinator_failed"],
            "urgency": case.final["urgency"]}
    return rows, cost


# ---------------- readable transcript ----------------
def render(path):
    lines = [json.loads(l) for l in open(path, encoding="utf-8")]
    case = lines[-1]["case"]
    out = [f"=== {case['post_id']} ===", case["post"][:1200] + ("..." if len(case["post"]) > 1200 else ""), ""]
    for m in lines[:-1]:
        o = m["output"] or {}
        head = f"[round {m['round']}] {m['agent']}  ({m['tokens_in']} in / {m['tokens_out']} out, {m['latency_s']}s" \
               f"{', INVALID' if not m['valid'] else ''})"
        out.append(head)
        if m["agent"] == "intake":
            out.append(f"   urgency {o.get('urgency')}: {o.get('urgency_reasons')}")
            out.append(f"   facts: {o.get('facts')}")
        elif m["agent"] == "router":
            out.append("   call: " + ", ".join(g for g, v in o.items() if v["call"]))
        elif m["agent"].startswith("specialist"):
            for c, v in o.items():
                if c == "consult_requests":
                    if v:
                        out.append(f"   consult requests: {v}")
                elif v["present"] or v["confidence"] < 0.7:
                    out.append(f"   {c}: {'PRESENT' if v['present'] else 'absent'} {v['confidence']:.2f} "
                               f"\"{v['evidence_quote'][:90]}\" | {v['rationale']}")
        elif m["agent"] == "skeptic":
            for ch in o.get("challenges", []):
                out.append(f"   challenge {ch['category']} [{ch['type']}]: {ch['argument']}")
            for ms in o.get("possible_misses", []):
                out.append(f"   possible miss {ms['category']}: {ms['reason']}")
        elif m["agent"].startswith("rebuttal"):
            out.append(f"   {o.get('decision')} -> present={o.get('present')} {o.get('new_confidence')}: {o.get('response')}")
        elif m["agent"] == "coordinator":
            out.append("   labels: " + ", ".join(c for c, v in o.get("labels", {}).items() if v["present"]))
            out.append(f"   urgency {o.get('urgency')}, settled {o.get('settled')}, reopen {o.get('reopen')}")
            out.append(f"   referral: {o.get('referral_note')}")
    skeptic = [d for d in case["debate"] if d["type"] == "skeptic"]
    if skeptic:
        out.append(f"(skeptic items kept: {[i['category'] for i in skeptic[0]['items']]}, dropped by tau/rules: {skeptic[0]['dropped']})")
    out.append("FINAL: " + ", ".join(c for c, v in case["final"]["labels"].items() if v) +
               f" | urgency {case['final']['urgency']} | rounds {case['final']['rounds']}")
    return "\n".join(out)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(render(p), "\n")

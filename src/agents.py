"""One function per agent. Each agent:
  1. builds its prompt from prompts/*.md, showing only the parts of the case file it needs,
  2. makes one LLM call with a strict JSON schema,
  3. writes its result into the case file and logs the message in the transcript.
"""
import hashlib

from src.evaluate import quote_in_post
from src.fewshot import few_shot_block
from src.prompts import category_definitions, load_prompt
from src.schemas import (CATEGORIES, GROUP_OF, GROUPS, SPECIALIST_SCHEMAS, B3Output, ClaimState, CoordinatorOutput,
                         Intake, Rebuttal, RouterOutput, SkepticMissesOutput, SkepticOutput)

NAMES = {"housing": "Housing", "family": "Family", "work_money": "Work & Money",
         "courts_crime": "Courts & Crime", "health_more": "Health & More", "general": "General"}

EVIDENCE_RETRY_NOTE = ("\n\nNote: your previous answer quoted text that is not in the post for: {cats}. "
                       "For these, copy words exactly as written in the post, or set present to false "
                       "if nothing in the post supports it.")


# ---------------- helpers ----------------
def call(llm, case, s, agent, round_, system, user, schema, model=None):
    """One LLM call, logged in the transcript. Returns the parsed dict, or None if invalid."""
    r = llm.chat(system, user, schema=schema, model=model)
    case.messages.append({
        "agent": agent, "round": round_,
        "prompt_hash": hashlib.sha256((system + user).encode()).hexdigest()[:12],
        "model": model or llm.model, "tokens_in": r.tokens_in, "tokens_out": r.tokens_out,
        "latency_s": round(r.latency, 2), "cached": r.cached, "attempts": r.attempts,
        "valid": r.valid, "output": r.parsed})
    if s.on_event:
        s.on_event(case.messages[-1], case)
    return r.parsed


def post_block(case):
    return f"Post:\n{case.post}"


def intake_block(case):
    i = case.intake or {}
    return ("Intake notes:\n- facts: " + "; ".join(i.get("facts", [])) +
            "\n- goals: " + "; ".join(i.get("user_goals", [])) +
            f"\n- state: {i.get('location_state')}")


def flagged(claim, tau):
    """A positive claim the skeptic may challenge: low confidence or evidence not in the post."""
    return claim.present and (claim.confidence < tau or claim.evidence_valid is False)


def claim_lines(case, tau=None):
    lines = []
    for c in CATEGORIES:
        cl = case.claims[c]
        if cl.specialist is None:
            lines.append(f"- {c}: not examined")
        elif cl.present:
            review = " [REVIEW]" if tau is not None and flagged(cl, tau) else ""
            lines.append(f'- {c}: PRESENT (confidence {cl.confidence:.2f}), evidence: "{cl.evidence_quote}", '
                         f"rationale: {cl.rationale}{review}")
        else:
            lines.append(f"- {c}: absent (confidence {cl.confidence:.2f})")
    return "\n".join(lines)


def set_claim(case, category, present, confidence, quote, rationale, specialist):
    case.claims[category] = ClaimState(
        present=present, confidence=confidence, evidence_quote=quote if present else "",
        evidence_valid=quote_in_post(quote, case.post) if present else None,
        rationale=rationale, specialist=specialist)


# ---------------- 1. Intake ----------------
def intake(llm, case, s):
    out = call(llm, case, s, "intake", 0, load_prompt("intake"), post_block(case), Intake)
    case.intake = out or {"facts": [], "parties": [], "location_state": None, "timeline": [],
                          "user_goals": [], "urgency": "medium", "urgency_reasons": ["intake failed"]}


# ---------------- 2. Router ----------------
def router(llm, case, s):
    if not s.use_router:  # ablation (b): call every specialist
        case.routing = {"specialists_to_call": list(GROUPS), "reasons": {g: "router disabled" for g in GROUPS}}
        return
    out = call(llm, case, s, "router", 0, load_prompt("router"),
               post_block(case) + "\n\n" + intake_block(case), RouterOutput)
    if out is None:  # favor recall: call everyone if the router fails
        out = {g: {"call": True, "reason": "router failed"} for g in GROUPS}
    to_call = [g for g in GROUPS if out[g]["call"]] or list(GROUPS)
    case.routing = {"specialists_to_call": to_call, "reasons": {g: out[g]["reason"] for g in GROUPS}}


# ---------------- 3. Specialists ----------------
def specialist(llm, case, s, group):
    """Round-0 judgment of one specialist. Returns its consult requests."""
    cats = CATEGORIES if group == "general" else GROUPS[group]
    others = ", ".join(f"{g} ({', '.join(GROUPS[g])})" for g in GROUPS if g != group) if group != "general" else "none"
    system = load_prompt("specialist", name=NAMES[group], categories=", ".join(cats),
                         definitions=category_definitions(cats), others=others)
    if s.few_shot:  # experiment (h)
        system += "\n\n" + few_shot_block(tuple(cats))
    user = post_block(case) + "\n\n" + intake_block(case)
    schema = SPECIALIST_SCHEMAS[group]
    out = call(llm, case, s, f"specialist:{group}", 0, system, user, schema, model=s.specialist_model)
    if out is None:
        return []

    # Evidence must be verbatim. Re-prompt once if any positive claim's quote is not in the post.
    bad = [c for c in cats if out[c]["present"] and not quote_in_post(out[c]["evidence_quote"], case.post)]
    n_pos = sum(out[c]["present"] for c in cats)
    case.evidence["first_try_total"] += n_pos
    case.evidence["first_try_valid"] += n_pos - len(bad)
    if bad:
        retry = call(llm, case, s, f"specialist:{group}:evidence_retry", 0, system,
                     user + EVIDENCE_RETRY_NOTE.format(cats=", ".join(bad)), schema, model=s.specialist_model)
        if retry is not None:
            out = retry

    case.specialist_outputs[group] = out
    for c in cats:
        cl = out[c]
        set_claim(case, c, cl["present"], cl["confidence"], cl["evidence_quote"], cl["rationale"], group)
    return out.get("consult_requests", [])


# ---------------- v2: Scout ----------------
def scout(llm, case, s):
    """One broad reading of all 16 categories (the same call as baseline B3).
    The specialists do not see it; only disagreements are raised, after they have judged."""
    case.scout = call(llm, case, s, "scout", 0, load_prompt("b3_single"), post_block(case), B3Output)


def scout_items(case, s):
    """Questions for the specialists where the Scout and a specialist disagree."""
    items = []
    for c in CATEGORIES if case.scout else []:
        sees, claim = case.scout[c], case.claims[c]
        if sees["present"] and not claim.present:
            items.append({"category": c, "kind": "scout_add", "from": "Scout",
                          "text": f'I think the post involves {c}. I noticed: "{sees["evidence_quote"]}"'})
        elif claim.present and not sees["present"] and s.scout_mode == "both":
            items.append({"category": c, "kind": "scout_drop", "from": "Scout",
                          "text": f"I read the post and did not see a {c} issue. Is it really there?"})
    return items


# ---------------- 4. Skeptic ----------------
def skeptic_misses(llm, case, s, round_):
    """v2: the skeptic raises possible misses only."""
    user = post_block(case) + "\n\nCurrent claims:\n" + claim_lines(case)
    out = call(llm, case, s, "skeptic", round_, load_prompt("skeptic_misses"), user, SkepticMissesOutput)
    items, seen, dropped = [], set(), 0
    for m in out["possible_misses"] if out else []:
        c = m["category"]
        if c in seen or case.claims[c].present:
            dropped += 1
            continue
        seen.add(c)
        items.append({"category": c, "kind": "possible_miss", "from": "Skeptic", "text": m["reason"]})
    case.debate.append({"round": round_, "type": "skeptic", "items": items, "dropped": dropped})
    return items


def skeptic(llm, case, s, round_):
    """Returns the items the specialists must answer: kept challenges and possible misses."""
    if s.skeptic_misses_only:
        return skeptic_misses(llm, case, s, round_)
    user = post_block(case) + "\n\nCurrent claims:\n" + claim_lines(case, tau=s.tau)
    out = call(llm, case, s, "skeptic", round_, load_prompt("skeptic", tau=s.tau), user, SkepticOutput)
    if out is None:
        return []
    items, seen = [], set()
    dropped = 0
    for ch in out["challenges"]:  # tau is enforced here, not only in the prompt
        c = ch["category"]
        if c in seen or not flagged(case.claims[c], s.tau):
            dropped += 1
            continue
        seen.add(c)
        items.append({"category": c, "kind": "challenge", "from": "Skeptic",
                      "text": f"[{ch['type']}] {ch['argument']}", "type": ch["type"]})
    for m in out["possible_misses"]:
        c = m["category"]
        if c in seen or case.claims[c].present:
            dropped += 1
            continue
        seen.add(c)
        items.append({"category": c, "kind": "possible_miss", "from": "Skeptic", "text": m["reason"]})
    case.debate.append({"round": round_, "type": "skeptic", "items": items, "dropped": dropped})
    return items


# ---------------- 5. Rebuttal ----------------
def owner(case, s, category):
    if s.single_specialist:
        return "general"
    return case.claims[category].specialist or GROUP_OF[category]


def rebuttal(llm, case, s, item, round_):
    c = item["category"]
    group = owner(case, s, c)
    cl = case.claims[c]
    if cl.specialist is None:
        prior = f'Your team did not examine "{c}" earlier; it is currently marked absent.'
    else:
        prior = (f'Your earlier judgment on "{c}": present={cl.present}, confidence {cl.confidence:.2f}, '
                 f'evidence: "{cl.evidence_quote}", rationale: {cl.rationale}')
    system = load_prompt("rebuttal", name=NAMES[group], category=c, definitions=category_definitions([c]))
    user = f"{post_block(case)}\n\n{prior}\n\nQuestion from the {item['from']}: {item['text']}"
    out = call(llm, case, s, f"rebuttal:{group}", round_, system, user, Rebuttal, model=s.specialist_model)
    if out is not None and out["present"] and not quote_in_post(out["evidence_quote"], case.post):
        retry = call(llm, case, s, f"rebuttal:{group}:evidence_retry", round_, system,
                     user + EVIDENCE_RETRY_NOTE.format(cats=c), Rebuttal, model=s.specialist_model)
        out = retry or out
    if out is None:
        return
    before = cl.present
    set_claim(case, c, out["present"], out["new_confidence"], out["evidence_quote"], out["response"], group)
    case.debate.append({"round": round_, "type": "rebuttal", "category": c, "kind": item["kind"],
                        "question": item["text"], "specialist": group, "decision": out["decision"],
                        "before": before, "after": out["present"], "response": out["response"]})


# ---------------- 6. Coordinator ----------------
def debate_lines(case):
    lines = []
    for d in case.debate:
        if d["type"] == "rebuttal":
            change = "unchanged" if d["before"] == d["after"] else f"changed to {'present' if d['after'] else 'absent'}"
            lines.append(f"- round {d['round']}, {d['category']}: {d['kind']} \"{d['question']}\" -> "
                         f"{d['specialist']} {d['decision']} ({change}): {d['response']}")
    return "\n".join(lines) or "(no debate)"


def coordinator(llm, case, s, round_, rounds_left):
    i = case.intake or {}
    user = (f"{post_block(case)}\n\nIntake: urgency {i.get('urgency')} "
            f"({'; '.join(i.get('urgency_reasons', []))}); goals: {'; '.join(i.get('user_goals', []))}\n\n"
            f"Current claims:\n{claim_lines(case)}\n\nDebate so far:\n{debate_lines(case)}\n\n"
            f"Rounds of questions left: {rounds_left}" + (" (you must set settled to true)" if rounds_left == 0 else ""))
    prompt = "coordinator_v2" if s.safe_note else "coordinator"
    out = call(llm, case, s, "coordinator", round_, load_prompt(prompt), user, CoordinatorOutput)
    case.decisions.append(out or {"invalid": True, "round": round_})
    return out

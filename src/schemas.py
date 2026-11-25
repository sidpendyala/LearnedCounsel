"""Pydantic models for every structured LLM output, and the shared case file.

Design rule from M2: use one FIXED field per category instead of open-ended lists,
and give every list a maximum length. With open lists the small model kept adding
items until it hit the token limit.
"""
from typing import Literal

from pydantic import BaseModel, Field, create_model

from src.data import load_config

_cfg = load_config()
CATEGORIES = _cfg["categories"]
GROUPS = _cfg["groups"]                       # specialist -> its categories
GROUP_OF = {c: g for g, cats in GROUPS.items() for c in cats}

Category = Literal[tuple(CATEGORIES)]
Specialist = Literal[tuple(GROUPS)]
Short = Field(max_length=200)                 # ~25-30 words


# ---------------- B3 ----------------
class CategoryAnswer(BaseModel):
    present: bool
    evidence_quote: str = Field(max_length=300)


B3Output = create_model("B3Output", **{c: (CategoryAnswer, ...) for c in CATEGORIES})


class CategoryAnswerEvidenceFirst(BaseModel):
    evidence_quote: str = Field(max_length=300)
    present: bool


B3EvidenceFirstOutput = create_model("B3EvidenceFirstOutput",
                                     **{c: (CategoryAnswerEvidenceFirst, ...) for c in CATEGORIES})


# ---------------- B2 ----------------
class B2Output(BaseModel):
    answer: Literal["Yes", "No"]


# ---------------- 1. Intake ----------------
class Intake(BaseModel):
    facts: list[str] = Field(max_length=8)
    parties: list[str] = Field(max_length=6)
    location_state: str | None
    timeline: list[str] = Field(max_length=6)
    user_goals: list[str] = Field(max_length=4)
    urgency: Literal["low", "medium", "high"]
    urgency_reasons: list[str] = Field(max_length=3)


# ---------------- 2. Router ----------------
class RouteDecision(BaseModel):
    call: bool
    reason: str = Field(max_length=200)


RouterOutput = create_model("RouterOutput", **{g: (RouteDecision, ...) for g in GROUPS})


# ---------------- 3. Specialists ----------------
class Claim(BaseModel):
    # Evidence and reasoning come BEFORE the decision, so the model explains first (M4 fix).
    evidence_quote: str = Field(max_length=300)
    rationale: str = Field(max_length=200)
    present: bool
    confidence: float = Field(ge=0, le=1)


class ConsultRequest(BaseModel):
    specialist: Specialist
    reason: str = Field(max_length=200)


def specialist_schema(name, categories):
    fields = {c: (Claim, ...) for c in categories}
    fields["consult_requests"] = (list[ConsultRequest], Field(max_length=2))
    return create_model(f"Specialist_{name}", **fields)


SPECIALIST_SCHEMAS = {g: specialist_schema(g, cats) for g, cats in GROUPS.items()}
SPECIALIST_SCHEMAS["general"] = specialist_schema("general", CATEGORIES)  # ablation (d)


# ---------------- 4. Skeptic ----------------
class Challenge(BaseModel):
    category: Category
    type: Literal["weak_evidence", "wrong_category", "overlap", "overclaim"]
    argument: str = Field(max_length=200)


class PossibleMiss(BaseModel):
    category: Category
    reason: str = Field(max_length=200)


class SkepticOutput(BaseModel):
    challenges: list[Challenge] = Field(max_length=6)
    possible_misses: list[PossibleMiss] = Field(max_length=4)


class SkepticMissesOutput(BaseModel):          # v2: the skeptic only looks for misses
    possible_misses: list[PossibleMiss] = Field(max_length=4)


# ---------------- 5. Rebuttal ----------------
class Rebuttal(BaseModel):
    # Reasoning first, decision last (M4 fix: a rebuttal said "not torts" but returned present=true).
    response: str = Field(max_length=200)
    evidence_quote: str = Field(max_length=300)
    decision: Literal["maintain", "withdraw", "revise"]
    present: bool
    new_confidence: float = Field(ge=0, le=1)


# ---------------- 6. Coordinator ----------------
class FinalLabel(BaseModel):
    present: bool
    confidence: float = Field(ge=0, le=1)


class ReopenItem(BaseModel):
    category: Category
    question: str = Field(max_length=200)


CoordinatorOutput = create_model(
    "CoordinatorOutput",
    labels=(create_model("FinalLabels", **{c: (FinalLabel, ...) for c in CATEGORIES}), ...),
    urgency=(Literal["low", "medium", "high"], ...),
    referral_note=(str, Field(max_length=900)),
    settled=(bool, ...),
    reopen=(list[ReopenItem], Field(max_length=3)),
)


# ---------------- B4: the same workflow in one context ----------------
class SelfChallenge(BaseModel):
    category: Category
    argument: str = Field(max_length=200)
    present: bool


B4Output = create_model(
    "B4Output",
    facts=(list[str], Field(max_length=8)),
    urgency=(Literal["low", "medium", "high"], ...),
    first_pass=(create_model("FirstPass", **{c: (Claim, ...) for c in CATEGORIES}), ...),
    challenges=(list[SelfChallenge], Field(max_length=6)),
    final_labels=(create_model("B4Labels", **{c: (bool, ...) for c in CATEGORIES}), ...),
    referral_note=(str, Field(max_length=900)),
)


# ---------------- The shared case file ----------------
class ClaimState(BaseModel):
    """Current state of one category, updated as the debate goes on."""
    present: bool = False
    confidence: float = 0.5
    evidence_quote: str = ""
    evidence_valid: bool | None = None
    rationale: str = ""
    specialist: str | None = None      # None = no specialist examined this category


class CaseFile(BaseModel):
    post_id: str
    post: str
    intake: dict | None = None
    routing: dict = {}                 # {"specialists_to_call": [...], "reasons": {...}}
    consults: list[dict] = []          # consult requests honored
    specialist_outputs: dict = {}      # round-0 output of each specialist
    scout: dict | None = None          # v2: first reading of all 16 categories
    claims: dict[str, ClaimState] = {c: ClaimState() for c in CATEGORIES}
    debate: list[dict] = []            # challenges, misses, rebuttals, reopen questions
    decisions: list[dict] = []         # coordinator output of each round
    final: dict | None = None          # final labels, confidences, urgency, referral note
    evidence: dict = {"first_try_valid": 0, "first_try_total": 0}
    messages: list[dict] = []          # transcript: one entry per LLM call

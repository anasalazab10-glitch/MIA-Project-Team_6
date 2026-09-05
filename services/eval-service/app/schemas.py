from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# -----------------------------
# Benchmark item (your 100-Q file)
# -----------------------------

class Scale(str, Enum):
    THOUSAND = "thousand"
    MILLION = "million"
    BILLION = "billion"
    PERCENT = "percent"
    NONE = "none"


class GoldEvidence(BaseModel):
    source_doc_uid: str | None = None
    source_page: int | None = None
    content_type: str | None = None
    gold_facts: list[str] | list[float] | None = None


class BenchmarkItem(BaseModel):
    @field_validator("scale", mode="before")
    @classmethod
    def coerce_scale(cls, v):
        # Accept '', None, 'none', 'percent', etc.
        if v is None:
            return Scale.NONE
        s = str(v).strip().lower()
        if s == "" or s == "none":
            return Scale.NONE
        if s == "percent":
            return Scale.PERCENT
        if s in ("thousand", "million", "billion"):
            return Scale(s)
        # Unknown scale -> treat as none (don’t crash eval)
        return Scale.NONE

    question_id: str
    question_text: str
    is_answerable: bool = True

    # Your benchmark label (e.g. "arithmetic") - not necessarily the strict answer schema type
    answer_type: str | None = None

    ground_truth_answer: str | int | float | list[str] | list[int] | list[float] | None = None
    scale: Scale = Scale.NONE

    gold_evidence: list[GoldEvidence] = Field(default_factory=list)


# -----------------------------
# Strict Answer schema (what your validator will enforce)
# -----------------------------

class EvidenceCitation(BaseModel):
    document_id: str
    page: list[int]
    section: str | None = None


class DirectParams(BaseModel):
    value: str | int | float


class CalculatedParams(BaseModel):
    value: float
    formula: str


class MultiSpanParams(BaseModel):
    values: list[str | int | float]


class InsufficientParams(BaseModel):
    reason: str


class AnswerType(str, Enum):
    DIRECT = "direct"
    CALCULATED = "calculated"
    MULTI_SPAN = "multi_span"
    INSUFFICIENT = "insufficient_evidence"


class StrictAnswer(BaseModel):
    answer_type: AnswerType
    evidence: list[EvidenceCitation] = Field(default_factory=list)
    params: dict[str, Any]


# -----------------------------
# Eval request/response
# -----------------------------

class RunBenchmarkRequest(BaseModel):
    orchestrator_url: str | None = None  # later: e.g. http://orchestrator:8000
    limit: int | None = Field(default=None, ge=1, le=100000)
    langfuse_project: str | None = None  # optional label


class PerItemResult(BaseModel):
    question_id: str
    question_text: str
    is_answerable: bool
    ground_truth_answer: Any
    predicted_answer: Any | None = None
    predicted_answer_type: str | None = None
    em: float | None = None
    f1: float | None = None
    numeric_ok: bool | None = None
    error: str | None = None


class RunBenchmarkResponse(BaseModel):
    num_items: int
    num_scored: int
    avg_em: float | None = None
    avg_f1: float | None = None
    numeric_accuracy: float | None = None
    results: list[PerItemResult]

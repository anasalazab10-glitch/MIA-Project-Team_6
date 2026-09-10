import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class DocumentCitation(BaseModel):
    document_id: str
    page: Union[int, List[int]] = [1]
    section: Optional[str] = "General"

    @field_validator("page", mode="before")
    @classmethod
    def coerce_page(cls, v):
        if isinstance(v, int):
            return [v]
        if isinstance(v, list):
            res = []
            for x in v:
                try:
                    res.append(int(x))
                except (ValueError, TypeError):
                    pass
            return res if res else [1]
        if isinstance(v, str):
            digits = re.findall(r"\d+", v)
            return [int(d) for d in digits] if digits else [1]
        return [1]


class RetrievedChunk(BaseModel):
    document_id: str
    page: Union[int, List[int]] = [1]
    section: Optional[str] = "General"
    content_type: str
    text: str
    chunk_id: Optional[str] = None
    score: Optional[float] = None
    rank: Optional[int] = None
    retrieval_method: Optional[str] = None
    scores: Optional[Dict[str, float]] = None

    @field_validator("page", mode="before")
    @classmethod
    def coerce_page(cls, v):
        if isinstance(v, int):
            return [v]
        if isinstance(v, list):
            res = []
            for x in v:
                try:
                    res.append(int(x))
                except (ValueError, TypeError):
                    pass
            return res if res else [1]
        if isinstance(v, str):
            digits = re.findall(r"\d+", v)
            return [int(d) for d in digits] if digits else [1]
        return [1]


# 1. Direct Schema
class DirectParams(BaseModel):
    value: Union[str, float, int]  # spec allows "string or number"


class DirectAnswer(BaseModel):
    answer_type: str = "direct"
    evidence: List[DocumentCitation] = Field(min_length=1)  # spec: at least 1 citation required
    params: DirectParams


# 2. Calculated Schema
class CalculatedParams(BaseModel):
    value: float
    formula: str


class CalculatedAnswer(BaseModel):
    answer_type: str = "calculated"
    evidence: List[DocumentCitation] = Field(min_length=1)
    params: CalculatedParams


# 3. Multi-Span Schema
class MultiSpanParams(BaseModel):
    values: List[str]


class MultiSpanAnswer(BaseModel):
    answer_type: str = "multi_span"
    evidence: List[DocumentCitation] = Field(min_length=1)
    params: MultiSpanParams


# 4. Insufficient Evidence Schema
class InsufficientEvidenceParams(BaseModel):
    reason: str


class InsufficientEvidenceAnswer(BaseModel):
    answer_type: str = "insufficient_evidence"
    evidence: List[DocumentCitation] = Field(default_factory=list)  # optional, may be empty
    params: InsufficientEvidenceParams
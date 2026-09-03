from typing import List, Optional,Union
from pydantic import BaseModel, Field


class DocumentCitation(BaseModel):
    document_id: str
    page: int
    section: Optional[str] = "General"


class RetrievedChunk(BaseModel):
    document_id: str
    page: int
    section: Optional[str] = "General"
    content_type: str
    text: str


# 1. Direct Schema
class DirectParams(BaseModel):
    value: Union[str, float, int]


class DirectAnswer(BaseModel):
    answer_type: str = "direct"
    evidence: List[DocumentCitation] = Field(min_length=1)   
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
    evidence: List[DocumentCitation] = Field(default_factory=list)
    params: InsufficientEvidenceParams
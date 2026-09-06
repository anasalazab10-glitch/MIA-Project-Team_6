import operator
from typing import Annotated, Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict

from schemas import DocumentCitation, RetrievedChunk


class AgentState(TypedDict):
    # Initial query inputs
    question: str
    session_id: Optional[str]

    # Classification & Routing
    question_type: Optional[Literal["direct", "calculated", "multi_span", "insufficient_evidence"]]
    is_cross_doc: bool
    sub_queries: List[Dict[str, str]]
    search_type: Literal["text", "table", "hybrid"]

    # Retrieval Context & Evidence
    retrieved_chunks: Annotated[List[RetrievedChunk], operator.add]
    evidence: List[DocumentCitation]
    evidence_status: Literal["sufficient", "weak", "insufficient"]
    retry_count: int

    # Arithmetic Execution
    formula: Optional[str]
    computed_value: Optional[float]

    # Reasoning / Output intermediate storage
    extracted_values: List[str]
    reasoning_summary: Optional[str]

    # Final formatted payload (matches schemas.py)
    final_answer: Optional[Dict[str, Any]]
"""
nodes/classifier.py

The first node in the reasoning pipeline. Reads the incoming question and
decides:
  - question_type: what kind of answer is expected (direct / calculated / multi_span)
  - search_type: whether to search text, tables, or both
  - is_cross_doc: whether the question likely needs more than one document
  - sub_queries: if the question should be broken into smaller search queries
"""
import json
import os
from typing import List, Optional

from groq_client import chat_completion_with_retry, MODEL_NAME
from pydantic import BaseModel, Field, field_validator, model_validator
from state import AgentState

class SubQuery(BaseModel):
    query: str
    purpose: str = "Search query"
    entity: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def parse_subquery(cls, v):
        if isinstance(v, str):
            return {"query": v, "purpose": "Search query"}
        return v


class ClassificationResult(BaseModel):
    question_type: str = "direct"
    search_type: str = "hybrid"
    is_cross_doc: bool = False
    companies: List[str] = Field(default_factory=list)
    sub_queries: List[SubQuery] = Field(default_factory=list)

    @field_validator("search_type", mode="before")
    @classmethod
    def coerce_search_type(cls, v):
        if not v or str(v).lower() not in ("text", "table", "hybrid"):
            return "hybrid"
        return str(v).lower()

    @field_validator("question_type", mode="before")
    @classmethod
    def coerce_question_type(cls, v):
        v = str(v).lower() if v else "direct"
        if v not in ("direct", "calculated", "multi_span", "insufficient_evidence"):
            return "direct"
        return v


CLASSIFIER_SYSTEM_PROMPT = """You are a question classifier for a financial document Q&A system.
Respond with ONLY valid JSON matching this exact schema:
{
  "question_type": "direct" | "calculated" | "multi_span" | "insufficient_evidence",
  "search_type": "text" | "table" | "hybrid",
  "is_cross_doc": true | false,
  "companies": ["Company A", "Company B"],
  "sub_queries": [ { "query": "...", "purpose": "...", "entity": "Company name if known" } ]
}

Rules for question_type:
- "direct": Asks for one single fact, metric, date, reason, definition, or explanation (e.g. "Why does X...", "What was revenue in 2020").
- "multi_span": Enumerate or list multiple distinct items, segments, components, or values for multiple periods without summing/averaging.
- "calculated": Requires arithmetic (total/sum over multiple periods or components, average, difference/distance, percentage change, ratio). E.g. "How far apart...", "average revenue", "total favourable impact".
- "insufficient_evidence": Never use for normal financial questions.

sub_queries: If the question mentions multiple companies or cross-document comparisons, create one sub_query per company with entity set to that company's name.
"""


def classify_question(state: AgentState) -> AgentState:
    """
    LangGraph node: reads state["question"], calls the LLM to classify it,
    and fills in question_type, search_type, is_cross_doc, sub_queries.
    """
    question = state["question"]

    response = chat_completion_with_retry(
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        response_format={"type": "json_object"},
        max_tokens=250,
        temperature=0,
    )

    raw_output = response.choices[0].message.content

    try:
        parsed = json.loads(raw_output)
        result = ClassificationResult(**parsed)
    except Exception as exc:
        # If the LLM ever returns malformed JSON, fail safe rather than crash
        # the whole pipeline - default to a hybrid direct search.
        print(f"[classifier] Failed to parse LLM output, using fallback. Error: {exc}")
        print(f"[classifier] Raw output was: {raw_output}")
        result = ClassificationResult(
            question_type="direct",
            search_type="hybrid",
            is_cross_doc=False,
            sub_queries=[],
        )

    state["question_type"] = result.question_type
    state["search_type"] = result.search_type
    state["is_cross_doc"] = result.is_cross_doc
    state["sub_queries"] = [sq.model_dump() for sq in result.sub_queries]
    state["companies"] = result.companies
    state["retry_count"] = 0  # initialize retry counter for the retriever node

    return state

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
- "calculated": Requires arithmetic, aggregation, or summing components. ANY question asking for "total", "sum", "average", "difference", "how far apart", "percentage change", "ratio", "change from ... to ...", or "total favourable/unfavourable impact" MUST be classified as "calculated".
  CRITICAL: If the question has a secondary clause like "Which page supports the answer?" or "Please cite the supporting page", ignore that clause for classification - classify based on what the main question asks!
- "multi_span": Enumerate or list multiple distinct items, segments, components, or values without summing/averaging (e.g. "Which components did X list under Y?").
- "direct": Asks for one single fact, metric, rate, date, reason, definition, or explanation that does not require arithmetic (e.g. "Why does X...", "What was revenue in 2020", "What is the tax rate...").
- "insufficient_evidence": Never use for normal financial questions.

Rules for search_type:
- Use "hybrid" for questions involving financial figures, line items, tables, or statements.

sub_queries: If the question mentions multiple companies or cross-document comparisons, create one sub_query per company with entity set to that company's name. For single-company questions, specify the company name in entity.
"""


def classify_question(state: AgentState) -> AgentState:
    """
    LangGraph node: reads state["question"], calls the LLM to classify it,
    and fills in question_type, search_type, is_cross_doc, sub_queries.
    """
    question = state["question"]

    try:
        response = chat_completion_with_retry(
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0,
        )
        raw_output = response.choices[0].message.content or "{}"
        raw_text = raw_output.strip()
        if "</think>" in raw_text:
            raw_text = raw_text.split("</think>")[-1].strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()
        if "{" in raw_text and "}" in raw_text:
            first_b = raw_text.find("{")
            last_b = raw_text.rfind("}")
            raw_text = raw_text[first_b : last_b + 1]
        parsed = json.loads(raw_text)
        result = ClassificationResult(**parsed)
    except Exception as exc:
        # If the LLM call fails or returns malformed JSON, fail safe rather than crash
        # the whole pipeline - default to a hybrid direct search.
        print(f"[classifier] Failed to call or parse LLM, using fallback. Error: {exc}")
        result = ClassificationResult(
            question_type="direct",
            search_type="hybrid",
            is_cross_doc=False,
            sub_queries=[],
        )

    # Deterministic guardrails for financial reasoning types:
    q_lower = question.lower()
    calc_keywords = [
        "total favourable impact", "total favorable impact",
        "how far apart", "absolute difference",
        "percentage change", "by what percentage", "growth rate",
        "average", "sum of", "as a percentage of", "as a % of",
        "total revenue between", "total equity for fiscal years",
        "change from 2018 to 2019", "change between 2018 and 2019",
        "change from 2017 to 2018", "change between 2017 and 2018",
        "average year-on-year", "sum of the three highest",
        "total net sales of the 3 highest", "top 3 components",
        "how many years between", "annual growth rate", "average total amount paid",
        "average operating income", "average risk-free interest rate",
    ]
    if any(kw in q_lower for kw in calc_keywords):
        if not (q_lower.startswith("why ") or "why was " in q_lower or "how was the " in q_lower):
            result.question_type = "calculated"
            result.search_type = "hybrid"

    multi_span_keywords = [
        "which components", "list under", "respectively",
        "which years", "which periods", "during which periods",
        "geographic regions", "what were the reasons", "what are the reasons",
        "what reasons did", "components make up", "which components did",
    ]
    if any(kw in q_lower for kw in multi_span_keywords):
        result.question_type = "multi_span"

    state["question_type"] = result.question_type
    state["search_type"] = result.search_type
    state["is_cross_doc"] = result.is_cross_doc
    state["sub_queries"] = [sq.model_dump() for sq in result.sub_queries]
    state["companies"] = result.companies
    state["retry_count"] = 0  # initialize retry counter for the retriever node

    return state

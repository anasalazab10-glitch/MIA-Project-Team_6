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

Given a user's question about financial reports, classify it and respond with ONLY valid JSON matching this exact schema:

{
  "question_type": "direct" | "calculated" | "multi_span" | "insufficient_evidence",
  "search_type": "text" | "table" | "hybrid",
  "is_cross_doc": true | false,
  "companies": ["Company A", "Company B"],
  "sub_queries": [ { "query": "...", "purpose": "...", "entity": "Company name if known" } ]
}

Rules for question_type:
- "direct": asks for one single fact, value, reason, or narrative explanation that can be directly found in the evidence.
  Examples:
  - "What was the revenue in 2020?"
  - "What was the cost of revenue in 2018?"
  - "Why does Company X expect to recognize deferred revenue?"
  - "How was the fair value of RSUs calculated?"

- "multi_span": asks for multiple distinct facts or values to be listed/enumerated directly from the evidence without arithmetic.
  Use this when the question asks to list values for multiple years/periods without summing or averaging them (e.g. "What were the revenues in 2017 and 2018 respectively?").
  ALSO use "multi_span" when the question asks to enumerate, list, or identify every item,
  component, category, segment, or element that belongs under a single label, heading, or
  row grouping in the evidence.
  Examples:
  - "What were the respective revenue values in 2017 and 2018?"
  - "What are the respective proportion of cost of revenue as a percentage of revenue in 2017 and 2018?"
  - "What were the revenue and operating income?"
  - "Which components did the company list under Due within one year?"
  - "What are the components of Accrued and Other Current Liabilities?"
  - "What are the geographic regions in which the Company operates?"
  IMPORTANT: If the question asks for a "total" or "sum" or "average" over multiple years, classify as "calculated", NOT "multi_span".

- "calculated": requires performing arithmetic using values from the evidence.
  Use this whenever the question asks for a derived numeric result such as:
  - total or sum over multiple periods/years or across multiple categories/components (e.g. "What was total revenue...", "What was the total favourable impact...")
  - average over multiple periods/years (e.g. "What was the average revenue over 2017 and 2018?")
  - difference or distance between two numbers (e.g. "How far apart were the 2019 balances...", "What is the difference between...")
  - percentage change or growth rate
  - count of items meeting a threshold
  - ratio or margin that must be calculated
  IMPORTANT: Even if the question includes a secondary question like "Which page supports the answer?", if the primary question asks for a total, sum, average, or difference, classify as "calculated".
  Examples:
  - "What was the total revenue between 2015 to 2019?"
  - "What was the total favourable impact foreign currency translation had on certain of their consolidated financial results? Which page supports the answer?"
  - "What was the percentage change in revenue from 2017 to 2018?"
  - "What is the difference between revenue in 2017 and 2018?"
  - "What was the average revenue over 2017 and 2018?"
  - "How far apart were the finished-goods balances reported by X and Y?"

- "insufficient_evidence": DO NOT use this for normal financial questions just because
  you do not know the answer yourself. The classifier does not have access to the documents
  or retrieved evidence, so it cannot determine whether evidence is sufficient.
  Use "insufficient_evidence" ONLY if the question itself is genuinely nonsensical,
  malformed, or impossible to interpret as a financial-document question.

IMPORTANT:
- Never classify a normal, understandable financial question as "insufficient_evidence".
- If the question asks for a reason, definition, explanation, or "why/how", classify it as "direct".
- Evidence sufficiency will be determined later by the retrieval stage.
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

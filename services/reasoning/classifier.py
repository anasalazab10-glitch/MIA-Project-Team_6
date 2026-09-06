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
from typing import List

from groq import Groq
from pydantic import BaseModel, Field

from state import AgentState

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL_NAME = "openai/gpt-oss-20b"

class SubQuery(BaseModel):
    query: str = Field(description="A focused search query for one piece of information")
    purpose: str = Field(description="Brief note on why this sub-query is needed")


class ClassificationResult(BaseModel):
    question_type: str = Field(
        description="One of: direct, calculated, multi_span, insufficient_evidence"
    )
    search_type: str = Field(description="One of: text, table, hybrid")
    is_cross_doc: bool = Field(
        description="True if the question likely needs evidence from more than one document"
    )
    sub_queries: List[SubQuery] = Field(
        default_factory=list,
        description="Broken-down search queries. Empty list if the question is simple enough to search directly.",
    )


CLASSIFIER_SYSTEM_PROMPT = """You are a question classifier for a financial document Q&A system.

Given a user's question about financial reports, classify it and respond with ONLY valid JSON matching this exact schema:

{
  "question_type": "direct" | "calculated" | "multi_span" | "insufficient_evidence",
  "search_type": "text" | "table" | "hybrid",
  "is_cross_doc": true | false,
  "sub_queries": [ { "query": "...", "purpose": "..." } ]
}

Rules for question_type:
- "direct": a single fact lookup (e.g. "What was the revenue in 2020?")
- "calculated": requires arithmetic - percentage change, sum, difference, ratio (e.g. "What was the % change in revenue?")
- "multi_span": expects a list of multiple items (e.g. "Which three expense categories increased?")
- "insufficient_evidence": only use this if the question is clearly unanswerable or nonsensical from context alone

Rules for search_type:
- "table": question is about specific numbers likely found in financial statements/tables
- "text": question is about narrative/qualitative information
- "hybrid": could be either, or you are unsure

Rules for sub_queries:
- If the question needs multiple distinct pieces of evidence (e.g. a "calculated" question comparing two years),
  break it into one sub_query per piece of evidence needed.
- If the question is a simple single lookup, return an empty list - the main question will be used as the search query directly.

Respond with ONLY the JSON object, no explanation, no markdown formatting."""


def classify_question(state: AgentState) -> AgentState:
    """
    LangGraph node: reads state["question"], calls the LLM to classify it,
    and fills in question_type, search_type, is_cross_doc, sub_queries.
    """
    question = state["question"]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        response_format={"type": "json_object"},
        temperature=0,  # deterministic classification, not creative
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
    state["retry_count"] = 0  # initialize retry counter for the retriever node

    return state

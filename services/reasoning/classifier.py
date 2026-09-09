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
- "direct": asks for one single fact or value that can be directly found in the evidence.
  Examples:
  - "What was the revenue in 2020?"
  - "What was the cost of revenue in 2018?"

- "multi_span": asks for multiple facts or values that can be directly found in the evidence.
  Use this when the question asks for values for multiple years, periods, categories, or entities,
  even if those values are percentages, ratios, or other numeric values.
  Examples:
  - "What were the respective revenue values in 2017 and 2018?"
  - "What are the respective proportion of cost of revenue as a percentage of revenue in 2017 and 2018?"
  - "What were the revenue and operating income?"
  IMPORTANT: If the requested values already appear in the document and no arithmetic is required,
  classify as "multi_span", NOT "calculated".

- "calculated": requires performing arithmetic using values from the evidence.
  Use this ONLY when the question explicitly asks for a derived result such as:
  - percentage change
  - growth rate
  - difference
  - sum
  - average
  - ratio that must be calculated
  - margin that must be calculated
  Examples:
  - "What was the percentage change in revenue from 2017 to 2018?"
  - "What is the difference between revenue in 2017 and 2018?"
  - "What was the average revenue over 2017 and 2018?"
  IMPORTANT: Merely asking for a percentage, proportion, ratio, or values from multiple years does NOT make a question "calculated".
  If the percentage/proportion/ratio is already explicitly stated in the evidence, use "multi_span".

- "insufficient_evidence": DO NOT use this for normal financial questions just because
  you do not know the answer yourself. The classifier does not have access to the documents
  or retrieved evidence, so it cannot determine whether evidence is sufficient.
  Use "insufficient_evidence" ONLY if the question itself is genuinely nonsensical,
  malformed, or impossible to interpret as a financial-document question.

IMPORTANT:
- Never classify a normal, understandable financial question as "insufficient_evidence"
  merely because you cannot answer it from the question alone.
- Your job is to classify the TYPE of information being requested, not whether the
  information exists in the documents.
- If the question asks for a fact, explanation, reason, definition, or statement that
  could reasonably appear in a financial report, classify it as "direct".
- Evidence sufficiency will be determined later by the retrieval stage.
"""


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

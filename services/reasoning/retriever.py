
import json
import os
from typing import List

from groq import Groq
from pydantic import BaseModel, Field

from state import AgentState
from schemas import DocumentCitation, RetrievedChunk
from tools import search_documents, search_tables

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL_NAME = "openai/gpt-oss-20b"

MAX_RETRIES = 2

class SufficiencyResult(BaseModel):
    status: str = Field(description="One of: sufficient, weak, insufficient")
    reason: str = Field(description="Brief explanation of the judgment")


SUFFICIENCY_SYSTEM_PROMPT = """You are checking whether retrieved evidence is enough to answer a question.

You will be given a QUESTION and a list of EVIDENCE chunks pulled from financial documents.

Judge the evidence and respond with ONLY valid JSON matching this schema:

{
  "status": "sufficient" | "weak" | "insufficient",
  "reason": "..."
}

Rules:
- "sufficient": the evidence clearly contains everything needed to answer the question directly.
- "weak": the evidence is somewhat related but missing a piece, unclear, or only partially relevant.
- "insufficient": the evidence is irrelevant or does not address the question at all.

Respond with ONLY the JSON object, no explanation outside the JSON, no markdown formatting."""


def _run_search(query: str, search_type: str, mock_mode: bool = False) -> List[RetrievedChunk]:
    if search_type == "table":
        return search_tables(query, mock_mode=mock_mode)
    else:
        # "text" and "hybrid" both go through search_documents,
        # since it already performs hybrid (semantic + keyword) search
        return search_documents(query, search_type=search_type, mock_mode=mock_mode)


def _rule_based_check(chunks: List[RetrievedChunk]) -> bool:
    """
    Fast, free check: did we get anything back at all?
    Returns True if chunks exist (passes rule-based check), False if empty.
    """
    return len(chunks) > 0


def _llm_relevance_check(question: str, chunks: List[RetrievedChunk]) -> SufficiencyResult:
    """
    Slower, smarter check: only runs when chunks exist, to judge whether
    they're actually relevant and complete enough to answer the question.
    """
    evidence_text = "\n".join(
        f"- [{c.document_id}, page {c.page}, {c.section}]: {c.text}" for c in chunks
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SUFFICIENCY_SYSTEM_PROMPT},
            {"role": "user", "content": f"QUESTION: {question}\n\nEVIDENCE:\n{evidence_text}"},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw_output = response.choices[0].message.content

    try:
        parsed = json.loads(raw_output)
        return SufficiencyResult(**parsed)
    except Exception as exc:
        print(f"[retriever] Failed to parse sufficiency check, defaulting to 'weak'. Error: {exc}")
        return SufficiencyResult(status="weak", reason="Sufficiency check failed to parse.")


def retrieve_evidence(state: AgentState, mock_mode: bool = False) -> AgentState:
    """
    LangGraph node: executes search based on the classifier's plan, collects
    evidence, and judges whether it's sufficient using the hybrid check.
    """
    question = state["question"]
    search_type = state.get("search_type", "hybrid")
    sub_queries = state.get("sub_queries", [])

    # If the classifier broke the question into sub-queries, search each one.
    # Otherwise, just search the main question directly.
    queries = [sq["query"] for sq in sub_queries] if sub_queries else [question]

    all_chunks: List[RetrievedChunk] = []
    for query in queries:
        chunks = _run_search(query, search_type, mock_mode=mock_mode)
        all_chunks.extend(chunks)

    if not _rule_based_check(all_chunks):
        state["retrieved_chunks"] = all_chunks
        state["evidence_status"] = "insufficient"
        state["evidence"] = []
        state["retry_count"] = state.get("retry_count", 0) + 1
        return state
    sufficiency = _llm_relevance_check(question, all_chunks)

    state["retrieved_chunks"] = all_chunks
    state["evidence_status"] = sufficiency.status
    state["reasoning_summary"] = sufficiency.reason

    state["evidence"] = [
        DocumentCitation(document_id=c.document_id, page=c.page, section=c.section)
        for c in all_chunks
    ]

    if sufficiency.status != "sufficient":
        state["retry_count"] = state.get("retry_count", 0) + 1

    return state


def should_retry(state: AgentState) -> bool:
  
    status = state.get("evidence_status")
    retries = state.get("retry_count", 0)
    return status != "sufficient" and retries < MAX_RETRIES

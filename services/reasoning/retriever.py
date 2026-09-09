import json
import os
from typing import List, Optional

from groq import Groq
from pydantic import BaseModel, Field

from state import AgentState
from schemas import DocumentCitation, RetrievedChunk
from tools import search_documents, search_tables


client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL_NAME = "openai/gpt-oss-20b"

MAX_RETRIES = 2


class SufficiencyResult(BaseModel):
    status: str = Field(
        description="One of: sufficient, weak, insufficient"
    )
    reason: str = Field(
        description="Brief explanation of the judgment"
    )


SUFFICIENCY_SYSTEM_PROMPT = """
You are an evidence sufficiency checker.

Given a question and retrieved document chunks, determine whether
the retrieved evidence is sufficient to answer the question accurately.

Return:
- sufficient: evidence directly supports the answer
- weak: evidence is partially relevant but may require another search
- insufficient: evidence does not contain enough information

Be concise.
"""


def _run_search(
    query: str,
    search_type: str,
    document_id: Optional[str] = None,
    mock_mode: bool = False,
    trace_id: Optional[str] = None,
) -> List[RetrievedChunk]:

    if search_type == "table":
        return search_tables(
            query,
            document_id=document_id,
            mock_mode=mock_mode,
        )

    else:
        return search_documents(
            query,
            search_type=search_type,
            document_id=document_id,
            mock_mode=mock_mode,
        )


def _rule_based_check(chunks: List[RetrievedChunk]) -> bool:
    return len(chunks) > 0


def _llm_relevance_check(
    question: str,
    chunks: List[RetrievedChunk],
) -> SufficiencyResult:

    evidence_text = "\n\n".join(
        [
            f"Document: {c.document_id}\n"
            f"Page: {c.page}\n"
            f"Content:\n{c.text}"
            for c in chunks
        ]
    )

    prompt = f"""
Question:
{question}

Retrieved evidence:
{evidence_text}

Determine whether the retrieved evidence is sufficient to answer
the question accurately.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SUFFICIENCY_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        content = response.choices[0].message.content

        if not content:
            return SufficiencyResult(
                status="weak",
                reason="The relevance checker returned no response.",
            )

        # Try to parse JSON response
        try:
            data = json.loads(content)
            return SufficiencyResult(
                status=data.get("status", "weak"),
                reason=data.get(
                    "reason",
                    "Evidence relevance could not be fully determined.",
                ),
            )

        except json.JSONDecodeError:
            content_lower = content.lower()

            if "sufficient" in content_lower:
                return SufficiencyResult(
                    status="sufficient",
                    reason=content,
                )

            if "insufficient" in content_lower:
                return SufficiencyResult(
                    status="insufficient",
                    reason=content,
                )

            return SufficiencyResult(
                status="weak",
                reason=content,
            )

    except Exception as e:
        return SufficiencyResult(
            status="weak",
            reason=f"Relevance check failed: {str(e)}",
        )


def retrieve_evidence(
    state: AgentState,
    mock_mode: bool = False,
) -> AgentState:

    question = state["question"]

    search_type = state.get(
        "search_type",
        "hybrid",
    )

    sub_queries = state.get(
        "sub_queries",
        [],
    )

    document_id = state.get(
        "document_id",
    )

    queries = (
        [sq["query"] for sq in sub_queries]
        if sub_queries
        else [question]
    )

    all_chunks: List[RetrievedChunk] = []

    # Search for evidence for every query
    for query in queries:

        chunks = _run_search(
            query,
            search_type=search_type,
            document_id=document_id,
            mock_mode=mock_mode,
        )

        all_chunks.extend(chunks)

    # Remove duplicate chunks
    unique_chunks = {}

    for chunk in all_chunks:
        unique_chunks[chunk.chunk_id] = chunk

    all_chunks = list(unique_chunks.values())

    # No evidence found
    if not _rule_based_check(all_chunks):

        state["retrieved_chunks"] = all_chunks

        state["evidence_status"] = "insufficient"

        state["evidence"] = []

        state["retry_count"] = state.get(
            "retry_count",
            0,
        ) + 1

        state["reasoning_summary"] = (
            "No relevant evidence was retrieved."
        )

        return state

    # Check whether evidence is sufficient
    sufficiency = _llm_relevance_check(
        question,
        all_chunks,
    )

    state["retrieved_chunks"] = all_chunks

    state["evidence_status"] = sufficiency.status

    state["reasoning_summary"] = sufficiency.reason

    state["evidence"] = [
        DocumentCitation(
            document_id=c.document_id,
            page=c.page,
            section=c.section,
        )
        for c in all_chunks
    ]

    return state


def should_retry(state: AgentState) -> str:

    evidence_status = state.get(
        "evidence_status",
        "insufficient",
    )

    retry_count = state.get(
        "retry_count",
        0,
    )

    if (
        evidence_status in ["weak", "insufficient"]
        and retry_count < MAX_RETRIES
    ):
        return "retry"

    return "continue"
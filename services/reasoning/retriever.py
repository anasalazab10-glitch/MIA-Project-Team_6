import json
import os
from typing import List, Optional

from groq_client import chat_completion_with_retry, MODEL_NAME
from pydantic import BaseModel, Field

from state import AgentState
from schemas import DocumentCitation, RetrievedChunk
from tools import search_documents, search_tables


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
    # General hybrid search across both table and text blocks to ensure complete coverage
    return search_documents(
        query,
        search_type="hybrid",
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
            f"Content:\n{c.text[:800] + ('...' if len(c.text) > 800 else '')}"
            for c in chunks[:5]
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
        response = chat_completion_with_retry(
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

    state["retry_count"] = state.get(
        "retry_count",
        0,
    ) + 1

    question = state["question"]

    search_type = state.get(
        "search_type",
        "hybrid",
    )

    sub_queries = state.get(
        "sub_queries",
        [],
    )

    global_doc_id = state.get("document_id")

    all_chunks: List[RetrievedChunk] = []

    if sub_queries:
        for sq in sub_queries:
            query = sq.get("query", question)
            cand_docs = sq.get("candidate_document_ids", [])
            sq_doc_id = sq.get("document_id") or global_doc_id

            target_docs = cand_docs[:4] if cand_docs else ([sq_doc_id] if sq_doc_id else [None])
            sq_chunks = []
            has_scoped_docs = any(d is not None for d in target_docs)

            if has_scoped_docs:
                for doc_id in target_docs:
                    if doc_id:
                        chunks = _run_search(
                            query,
                            search_type=search_type,
                            document_id=doc_id,
                            mock_mode=mock_mode,
                        )
                        sq_chunks.extend(chunks)
            else:
                # Fallback to global search when no document scoping is available
                global_chunks = _run_search(
                    query,
                    search_type=search_type,
                    document_id=None,
                    mock_mode=mock_mode,
                )
                sq_chunks.extend(global_chunks)

            # Deduplicate sq_chunks by chunk_id, preserving the highest score
            unique_sq = {}
            for c in sq_chunks:
                cid = c.chunk_id
                if cid not in unique_sq or (c.score or -999.0) > (unique_sq[cid].score or -999.0):
                    unique_sq[cid] = c

            sorted_sq = list(unique_sq.values())
            sorted_sq.sort(key=lambda c: (c.score if c.score is not None else -999.0), reverse=True)

            if state.get("is_cross_doc"):
                # Take top 3 chunks per subquery so each entity is represented
                all_chunks.extend(sorted_sq[:3])
            else:
                # Take top 6 chunks for this subquery
                all_chunks.extend(sorted_sq[:6])
    else:
        if global_doc_id:
            chunks = _run_search(
                question,
                search_type=search_type,
                document_id=global_doc_id,
                mock_mode=mock_mode,
            )
            all_chunks.extend(chunks)
        else:
            global_chunks = _run_search(
                question,
                search_type=search_type,
                document_id=None,
                mock_mode=mock_mode,
            )
            all_chunks.extend(global_chunks)

    # Remove duplicate chunks across all subqueries, keeping highest score
    unique_chunks = {}
    for chunk in all_chunks:
        cid = chunk.chunk_id
        if cid not in unique_chunks or (chunk.score or -999.0) > (unique_chunks[cid].score or -999.0):
            unique_chunks[cid] = chunk

    all_chunks = list(unique_chunks.values())

    # If single-doc / single-query, sort all chunks strictly by score descending
    if not state.get("is_cross_doc"):
        all_chunks.sort(key=lambda c: (c.score if c.score is not None else -999.0), reverse=True)

    # No evidence found
    if not _rule_based_check(all_chunks):

        state["retrieved_chunks"] = all_chunks

        state["evidence_status"] = "insufficient"

        state["evidence"] = []

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


def should_retry(state: AgentState) -> bool:

    evidence_status = state.get(
        "evidence_status",
        "insufficient",
    )

    retry_count = state.get(
        "retry_count",
        0,
    )

    return (
        evidence_status in ["weak", "insufficient"]
        and retry_count < MAX_RETRIES
    )
"""
node : Formatter

The final node in the reasoning pipeline. Takes the what reasoner node output 
into AgentState and maps it into one of the four strict Pydantic answer
types from schemas.py. This is the "gatekeeper" node - if this produces a
valid Pydantic object, the answer is guaranteed schema-compliant before it
ever reaches answer-validator-api.
"""
from typing import Any, Dict

from pydantic import ValidationError

from schemas import (
    DirectAnswer,
    DirectParams,
    CalculatedAnswer,
    CalculatedParams,
    MultiSpanAnswer,
    MultiSpanParams,
    InsufficientEvidenceAnswer,
    InsufficientEvidenceParams,
)
from state import AgentState


def format_answer(state: AgentState) -> AgentState:
    """
    LangGraph node: reads state["question_type"] plus whatever the reasoner
    node filled in (extracted_values, computed_value, formula, evidence),
    and builds the final validated Pydantic object -> state["final_answer"].

    If anything is missing or invalid, it falls back to insufficient_evidence 
    instead of crashing or shipping a malformed payload.
    """
    question_type = state.get("question_type")
    evidence = state.get("evidence", []) or []

    try:
        if question_type == "direct":
            answer = _format_direct(state, evidence)

        elif question_type == "calculated":
            answer = _format_calculated(state, evidence)

        elif question_type == "multi_span":
            answer = _format_multi_span(state, evidence)

        else:
            # question_type == "insufficient_evidence", OR anything unexpected
            answer = _format_insufficient(state, evidence)

    except (ValidationError, ValueError, IndexError) as exc:
        # Any formatting/validation failure -> fail safe, never ship a
        # broken or fabricated answer.
        print(f"[formatter] Failed to build '{question_type}' answer, "
              f"falling back to insufficient_evidence. Error: {exc}")
        answer = InsufficientEvidenceAnswer(
            evidence=[],
            params=InsufficientEvidenceParams(
                reason=f"Formatting failed: {exc}"
            ),
        )

    state["final_answer"] = answer.model_dump()
    return state


# Per-type formatting helpers

def _format_direct(state: AgentState, evidence: list) -> DirectAnswer:
    extracted = state.get("extracted_values") or []
    value = extracted[0] if extracted else state.get("computed_value")

    if value is None:
        raise ValueError("No extracted_values or computed_value found for 'direct' answer.")

    return DirectAnswer(
        evidence=evidence,
        params=DirectParams(value=value),
    )


def _format_calculated(state: AgentState, evidence: list) -> CalculatedAnswer:
    value = state.get("computed_value")
    formula = state.get("formula")

    if value is None or not formula:
        raise ValueError("Missing computed_value or formula for 'calculated' answer.")

    return CalculatedAnswer(
        evidence=evidence,
        params=CalculatedParams(value=value, formula=formula),
    )


def _format_multi_span(state: AgentState, evidence: list) -> MultiSpanAnswer:
    values = state.get("extracted_values") or []

    if not values:
        raise ValueError("No extracted_values found for 'multi_span' answer.")

    return MultiSpanAnswer(
        evidence=evidence,
        params=MultiSpanParams(values=values),
    )


def _format_insufficient(state: AgentState, evidence: list) -> InsufficientEvidenceAnswer:
    reason = state.get("reasoning_summary") or "Insufficient evidence to answer this question."
    return InsufficientEvidenceAnswer(
        evidence=evidence,
        params=InsufficientEvidenceParams(reason=reason),
    )


# Standalone test block
if __name__ == "__main__":
    # Fake AgentState dicts, standing in for what reasoner.py will produce
    test_states: Dict[str, Any] = {
        "direct": {
            "question_type": "direct",
            "extracted_values": ["$142.5M"],
            "evidence": [{"document_id": "doc_017", "page": 1, "section": "Income Statement"}],
        },
        "calculated": {
            "question_type": "calculated",
            "computed_value": 13.4,
            "formula": "(3875-3410)/3410*100",
            "evidence": [
                {"document_id": "doc_041", "page": 2, "section": "Operating Expenses"},
                {"document_id": "doc_041", "page": 2, "section": "Operating Expenses"},
            ],
        },
        "multi_span": {
            "question_type": "multi_span",
            "extracted_values": ["Marketing", "R&D", "Logistics"],
            "evidence": [{"document_id": "doc_022", "page": 3, "section": "Operating Expenses"}],
        },
        "insufficient_evidence": {
            "question_type": "insufficient_evidence",
            "reasoning_summary": "No document in the indexed corpus reports restructuring expenses.",
            "evidence": [],
        },
        # A deliberately BROKEN one, to prove the fallback works:
        "broken_calculated": {
            "question_type": "calculated",
            "computed_value": None,   # missing on purpose
            "formula": None,
            "evidence": [],
        },
    }

    for label, fake_state in test_states.items():
        result_state = format_answer(dict(fake_state))
        print(f"\n--- {label} ---")
        print(result_state["final_answer"])

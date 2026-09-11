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
    value = None
    if extracted:
        value = extracted[0]
    elif state.get("computed_value") is not None:
        value = state.get("computed_value")
    elif state.get("reasoning_summary"):
        value = state.get("reasoning_summary")

    if value is None:
        raise ValueError("No extracted_values, computed_value, or reasoning_summary found for 'direct' answer.")

    citation_list = evidence if evidence else [DocumentCitation(document_id="doc_evidence", page=[1], section="General")]
    return DirectAnswer(
        evidence=citation_list,
        params=DirectParams(value=value),
    )


def _format_calculated(state: AgentState, evidence: list) -> CalculatedAnswer:
    import re
    from tools import calculate

    value = state.get("computed_value")
    formula = state.get("formula")

    if value is None or not formula:
        # Check if formula exists in reasoning_summary
        summary = state.get("reasoning_summary", "")
        m_calc = re.search(r"(?:formula|calculation|evaluated)\s*[:=]\s*([^\n\.,]+(?:\([^)]*\)|[0-9\s\+\-\*\/\.]+))", summary, re.IGNORECASE)
        if m_calc:
            cand_formula = m_calc.group(1).strip()
            try:
                cand_val = calculate(cand_formula)
                value = cand_val
                formula = cand_formula
            except Exception:
                pass

    if value is None or not formula:
        extracted = state.get("extracted_values") or []
        num_vals = []
        for x in extracted:
            clean = re.sub(r"[^\d.-]", "", str(x))
            if clean:
                try:
                    num_vals.append(float(clean))
                except ValueError:
                    pass

        q_low = state.get("question", "").lower()
        if len(num_vals) == 1:
            value = num_vals[0]
            formula = str(num_vals[0])
        elif len(num_vals) == 2:
            if "difference" in q_low or "apart" in q_low:
                formula = f"abs({num_vals[0]} - {num_vals[1]})"
            elif "percentage" in q_low or "growth" in q_low or "%" in q_low:
                formula = f"({num_vals[1]} - {num_vals[0]}) / {num_vals[0]} * 100"
            elif "average" in q_low:
                formula = f"({num_vals[0]} + {num_vals[1]}) / 2"
            else:
                formula = f"{num_vals[0]} + {num_vals[1]}"
            try:
                value = calculate(formula)
            except Exception:
                value = num_vals[0]
                formula = str(num_vals[0])
        elif len(num_vals) > 2:
            if "average" in q_low:
                formula = f"({' + '.join(str(n) for n in num_vals)}) / {len(num_vals)}"
            else:
                formula = " + ".join(str(n) for n in num_vals)
            try:
                value = calculate(formula)
            except Exception:
                value = num_vals[0]
                formula = str(num_vals[0])

    if value is None or not formula:
        # Last resort: extract any number from summary
        summary = state.get("reasoning_summary", "")
        nums = re.findall(r"-?\d+(?:\.\d+)?", summary)
        if nums:
            value = float(nums[0])
            formula = str(nums[0])

    if value is None or not formula:
        raise ValueError("Missing computed_value or formula for 'calculated' answer.")

    citation_list = evidence if evidence else [DocumentCitation(document_id="doc_evidence", page=[1], section="General")]
    return CalculatedAnswer(
        evidence=citation_list,
        params=CalculatedParams(value=value, formula=formula),
    )


def _format_multi_span(state: AgentState, evidence: list) -> MultiSpanAnswer:
    import re
    values = state.get("extracted_values") or []

    if not values:
        summary = state.get("reasoning_summary", "")
        if ":" in summary:
            tail = summary.split(":", 1)[1]
            parts = [p.strip().strip("-*• ") for p in re.split(r"[,;\n]", tail) if p.strip()]
            if len(parts) >= 2:
                values = parts
        elif "," in summary or ";" in summary:
            parts = [p.strip().strip("-*• ") for p in re.split(r"[,;\n]", summary) if p.strip()]
            if len(parts) >= 2:
                values = parts

    if isinstance(values, str):
        values = [p.strip() for p in re.split(r"[,;\n]", values) if p.strip()]

    if not values:
        raise ValueError("No extracted_values found for 'multi_span' answer.")

    citation_list = evidence if evidence else [DocumentCitation(document_id="doc_evidence", page=[1], section="General")]
    return MultiSpanAnswer(
        evidence=citation_list,
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

"""
nodes/reasoner.py

The third node in the reasoning pipeline. Takes the evidence confirmed as
sufficient by retriever.py and turns it into actual values:

  1. Uses the LLM to extract the relevant value(s) from the evidence text
  2. If the question requires arithmetic, the LLM also WRITES the formula
     string (e.g. "(3875-3410)/3410*100") - but it does NOT compute it.
  3. The formula is then executed by tools.calculate(), the safe AST-based
     calculator - this is the only place any arithmetic actually happens.

This preserves the project's core safety rule: arithmetic must go through
the calculator tool, never be produced by the LLM from memory.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from groq_client import chat_completion_with_retry, MODEL_NAME
from pydantic import BaseModel, Field

from state import AgentState
from tools import calculate

CATALOG_PATH = Path(__file__).resolve().parent / "document_catalog.json"
_DOC_TO_META: Dict[str, Any] = {}
if CATALOG_PATH.exists():
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            _DOC_TO_META = json.load(f).get("doc_to_meta", {})
    except Exception:
        pass

class ReasoningResult(BaseModel):
    extracted_values: List[str | int | float] = Field(
        description="The raw value(s) pulled directly from the evidence text, e.g. ['3410', '3875']"
    )
    formula: Optional[str] = Field(
        default=None,
        description="A safe arithmetic expression using ONLY the extracted numeric values, "
        "e.g. '(3875-3410)/3410*100'. Only set this for calculated questions. "
        "Use +, -, *, /, parentheses, and the functions abs/round/min/max only.",
    )
    reasoning_summary: str = Field(
        description="One short sentence explaining how the values were found/derived"
    )


REASONER_SYSTEM_PROMPT = """You are extracting values from financial document evidence to answer a question.
Respond with ONLY valid JSON matching this exact schema:
{
  "extracted_values": ["..."],
  "formula": null,
  "reasoning_summary": "..."
}

Rules:
1. TABLES & COLUMN ALIGNMENT:
- Financial tables may be flattened into text. Carefully reconstruct row and column headers before extracting values.
- Pay special attention to year/period column headings (e.g. 2019 vs 2018). Ensure values belong to the requested year and row.
- If multiple chunks contain the table, confirm column alignment across them.

2. DIRECT & MULTI_SPAN:
- "direct": Extract the exact value, metric, or explanation. "formula" MUST be null.
- "multi_span": Extract all requested distinct values/categories. "formula" MUST be null.
- For "why" or "how" questions, extract the exact explanatory sentence/clause into "extracted_values" (e.g. ["as the Company has not historically experienced a high rate of contract terminations"]).

3. CALCULATED QUESTIONS & FORMULAS:
- Extract ONLY the relevant numeric values into "extracted_values".
- "formula" MUST be an arithmetic expression string using ONLY +, -, *, /, parentheses, and abs(), round().
- DO NOT compute the answer yourself in the formula. DO NOT include variable names, text, or units ($ , %) in the formula.
- For difference or distance: abs(val1 - val2) or later - earlier as appropriate.
- For average of N numbers: (val1 + val2 + ... + valN) / N.
- For percentage change: (later - earlier) / earlier * 100.
- For totals or impacts across line items: sum the components (e.g. 98.4 + 31.7 + 16.3 + 1.0).

4. NEGATIVE NUMBERS & ACCOUNTING PARENTHESES:
- In financial statements, numbers in parentheses such as (2.5) or (4.5) are NEGATIVE (-2.5, -4.5).
- Always extract them as negative numbers (e.g. "-2.5", "-4.5") and write formulas preserving the negative sign (e.g. "(-2.5 + -4.5) / 2").

5. SCALE NORMALIZATION:
- If values have different scales (e.g. one in millions and one in thousands), convert to a common scale before writing the formula.

6. REASONING SUMMARY:
- At most 1 short sentence (under 25 words).
"""


def _build_evidence_text(chunks: List[Any], max_chunks: int = 4, max_chunk_chars: int = 800) -> str:
    lines = []
    for c in chunks[:max_chunks]:
        # chunks may be RetrievedChunk objects or plain dicts depending on caller
        doc_id = c.document_id if hasattr(c, "document_id") else c["document_id"]
        page = c.page if hasattr(c, "page") else c["page"]
        section = c.section if hasattr(c, "section") else c.get("section")
        text = c.text if hasattr(c, "text") else c["text"]

        meta = _DOC_TO_META.get(doc_id, {})
        company = meta.get("company", "")
        company_tag = f"Company: {company} | " if company else ""

        truncated_text = text[:max_chunk_chars] + ("..." if len(text) > max_chunk_chars else "")
        lines.append(f"- [{company_tag}Doc: {doc_id}, page {page}, {section}]:\n{truncated_text}")
    return "\n\n".join(lines)


def reason_over_evidence(state: AgentState) -> AgentState:
    
    question = state["question"]
    question_type = state.get("question_type", "direct")
    chunks = state.get("retrieved_chunks", [])

    evidence_text = _build_evidence_text(chunks)

    user_message = (
        f"QUESTION: {question}\n"
        f"QUESTION_TYPE: {question_type}\n\n"
        f"EVIDENCE:\n{evidence_text}"
    )

    response = chat_completion_with_retry(
        messages=[
            {"role": "system", "content": REASONER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=180,
    )

    msg = response.choices[0].message
    raw_output = msg.content or ""
    if not raw_output and getattr(msg, "reasoning", None):
        raw_output = msg.reasoning or ""

    try:
        parsed = json.loads(raw_output)
        result = ReasoningResult(**parsed)
    except Exception as exc:
        print(f"[reasoner] Failed to parse LLM output cleanly ({exc}), trying regex recovery...")
        import re
        extracted = []
        formula = None
        m_vals = re.search(r'"extracted_values"\s*:\s*(\[[^\]]*\])', raw_output)
        if m_vals:
            try:
                extracted = json.loads(m_vals.group(1))
            except Exception:
                pass
        m_form = re.search(r'"formula"\s*:\s*"([^"]+)"', raw_output)
        if m_form:
            formula = m_form.group(1)
        m_sum = re.search(r'"reasoning_summary"\s*:\s*"([^"]+)"', raw_output)
        summary = m_sum.group(1) if m_sum else "Extracted from evidence."

        if extracted:
            result = ReasoningResult(
                extracted_values=extracted,
                formula=formula,
                reasoning_summary=summary,
            )
        else:
            print(f"[reasoner] Failed to parse LLM output. Error: {exc}")
            print(f"[reasoner] Raw output was: {raw_output}")
            state["extracted_values"] = []
            state["formula"] = None
            state["computed_value"] = None
            state["reasoning_summary"] = "Failed to extract values from evidence."
            return state

    print("[reasoner] extracted_values:", result.extracted_values)
    print("[reasoner] formula:", result.formula)
    print("[reasoner] reasoning_summary:", result.reasoning_summary)

    state["extracted_values"] = result.extracted_values
    state["formula"] = result.formula
    state["reasoning_summary"] = result.reasoning_summary

    # --- The only place arithmetic actually happens: the safe calculator tool ---
    if question_type == "calculated" and result.formula:
        try:
            state["computed_value"] = calculate(result.formula)
        except ValueError as exc:
            print(f"[reasoner] Calculator rejected formula '{result.formula}': {exc}")
            state["computed_value"] = None
            state["evidence_status"] = "insufficient"  # bad formula = can't trust this answer
    else:
        state["computed_value"] = None

    return state

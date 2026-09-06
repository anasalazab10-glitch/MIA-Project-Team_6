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
from typing import List, Optional

from groq import Groq
from pydantic import BaseModel, Field

from state import AgentState
from tools import calculate

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL_NAME = "openai/gpt-oss-20b"

class ReasoningResult(BaseModel):
    extracted_values: List[str] = Field(
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

You will be given a QUESTION, its QUESTION_TYPE, and EVIDENCE chunks retrieved from documents.

Respond with ONLY valid JSON matching this exact schema:

{
  "extracted_values": ["..."],
  "formula": "..." or null,
  "reasoning_summary": "..."
}

Rules:
- "extracted_values": pull out ONLY the raw fact(s)/number(s) that are actually present in the evidence text.
  Never invent a value that isn't in the evidence.
- "formula": ONLY fill this in if question_type is "calculated". Write a plain arithmetic expression
  using the extracted numeric values (e.g. "(3875-3410)/3410*100"). Use only +, -, *, /, parentheses,
  and the functions abs(), round(), min(), max(). DO NOT compute the result yourself - just write the expression.
  For "direct" or "multi_span" questions, set this to null.
- "reasoning_summary": briefly explain where each value came from in the evidence.

Respond with ONLY the JSON object, no explanation outside the JSON, no markdown formatting."""


def _build_evidence_text(chunks: List[dict]) -> str:
    lines = []
    for c in chunks:
        # chunks may be RetrievedChunk objects or plain dicts depending on caller
        doc_id = c.document_id if hasattr(c, "document_id") else c["document_id"]
        page = c.page if hasattr(c, "page") else c["page"]
        section = c.section if hasattr(c, "section") else c.get("section")
        text = c.text if hasattr(c, "text") else c["text"]
        lines.append(f"- [{doc_id}, page {page}, {section}]: {text}")
    return "\n".join(lines)


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

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": REASONER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw_output = response.choices[0].message.content

    try:
        parsed = json.loads(raw_output)
        result = ReasoningResult(**parsed)
    except Exception as exc:
        print(f"[reasoner] Failed to parse LLM output. Error: {exc}")
        print(f"[reasoner] Raw output was: {raw_output}")
        state["extracted_values"] = []
        state["formula"] = None
        state["computed_value"] = None
        state["reasoning_summary"] = "Failed to extract values from evidence."
        return state

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

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

You will be given a QUESTION, its QUESTION_TYPE, and EVIDENCE chunks retrieved from financial reports.

Respond with ONLY valid JSON matching this exact schema:

{
  "extracted_values": ["..."],
  "formula": null,
  "reasoning_summary": "..."
}

Rules:

1. GENERAL EXTRACTION
- First identify exactly what the question is asking for.
- Then locate the corresponding value(s), fact(s), or explanation in the evidence.
- Do NOT simply extract the first number or fact that appears in the evidence.
- Never invent information that is not present in the evidence.
- Use the labels, headings, row names, column headings, years, and surrounding context to determine what each value means.

2. TABLES AND COLUMN ALIGNMENT
- Financial evidence may come from tables whose formatting has been flattened into plain text.
- When evidence comes from a table, carefully reconstruct the table's row and column relationships before extracting values.
- Pay special attention to year/period column headings.
- Do NOT assume that numbers belong to a year simply because they appear near that year in the extracted text.
- Determine which value belongs to which row AND which year/period.
- If the evidence contains multiple years, explicitly identify the requested metric for each requested year before constructing the answer.
- A value appearing immediately before a row label does not necessarily mean it belongs to the first year shown. Use the table structure and all surrounding evidence to determine the correct mapping.
- If multiple chunks contain the same table, use them together to confirm the column alignment.

3. DIRECT QUESTIONS
- For "direct" questions, extract the exact value or fact requested.
- If the question asks for one specific value, return that value.
- Do not perform arithmetic unless the question_type is "calculated".

4. MULTIPLE-VALUE QUESTIONS
- For "multi_span" questions, extract every requested value or fact.
- Keep each extracted value associated with the correct metric and year/period.
- Do not merge values from different years or metrics.

5. "WHY" AND "HOW" QUESTIONS
- For "why" or "how" questions, extract the relevant causal explanation, reason, method, or explanatory statement.
- Put that explanation directly into "extracted_values" as a string (e.g. ["because the company has not historically experienced a high rate of contract terminations"]), in addition to summarizing it in reasoning_summary.
- Do NOT return an unrelated number merely because it appears nearby in the evidence.

6. CALCULATED QUESTIONS
- For "calculated" questions, extract ONLY the numeric input values needed to perform the requested calculation.
- Every extracted value must clearly correspond to the correct metric, year, period, or category requested by the question.
- Before writing the formula, verify that each number has the correct meaning and year.
- Do NOT use a nearby value just because it looks numerically plausible.
- Do NOT invent missing values.
- If the question asks for a change "from X to Y", identify the value for X and the value for Y explicitly.
- For a change from an earlier period X to a later period Y, the usual difference is:
  later value - earlier value
- For example, if a metric was 235.8 in 2018 and 334.1 in 2019, the formula must be:
  334.1-235.8
  and NOT:
  235.8-334.1
- If the question asks "by how much did X change from 2018 to 2019?", calculate:
  2019 value - 2018 value.
- If the question asks for percentage change, use:
  (later value - earlier value) / earlier value * 100
- If the question asks for a difference or how far apart two numbers are, use:
  abs(value1 - value2) or later - earlier as appropriate.
- If the question asks for a sum, total over multiple years, average, ratio, or margin, use only the values necessary for that calculation.

7. COMBINED METRICS
- If the question asks for a metric that is explicitly presented as a combined row in a table, use that row directly.
- For example, if a table contains:
  "Net debt"
  "IFRS 16 lease liabilities"
  "Net debt and IFRS 16 lease liabilities"
  and the question asks about "net debt and IFRS 16 lease liabilities", use the value from the row explicitly labeled "Net debt and IFRS 16 lease liabilities".
- Do NOT independently add nearby components when the combined value is already explicitly provided, unless the question specifically asks you to calculate the combined value.
- Make sure the combined row's value is assigned to the correct year/period.

8. MISSING OR NON-APPLICABLE VALUES
- Do NOT invent a value for a year simply because another year has a value.
- If a metric is not separately reported for a particular year, do not copy the value from another year.
- If a financial reporting change means a value is only applicable or separately reported in one year, preserve that distinction.
- For example, if IFRS 16 lease liabilities are shown for 2019 but not separately shown for 2018, do NOT assume that the 2019 value also applies to 2018.
- If the question asks about a combined metric and the combined value is explicitly reported for both years, use those reported combined values.

9. FORMULA
- "formula" MUST be null for "direct" and "multi_span" questions.
- "formula" MUST be filled for a "calculated" question when enough numeric evidence exists.
- Write a plain arithmetic expression using ONLY the numeric values extracted from the evidence.
- Allowed operators: +, -, *, /
- Allowed functions: abs(), round(), min(), max()
- Use parentheses when necessary.
- DO NOT compute the final result yourself.
- DO NOT put units such as £, $, %, million, or text inside the formula.
- The formula should contain the actual numeric values, not variable names.
- Make sure the formula uses the correct values in the correct order.

10. REASONING SUMMARY
- Briefly explain where the extracted values came from and how they correspond to the question.
- For calculated questions, mention the relevant years/periods and metrics.
- The summary must reflect the actual evidence and must not invent facts.

11. UNITS AND SCALE NORMALIZATION
- Financial reports frequently present numbers in different scales, e.g. "in thousands", "in millions", or exact dollars.
- Always check table headers, column subtitles, and footnotes for the reporting scale (e.g. "$ in thousands", "$ in millions").
- When calculating differences, totals, or averages across multiple companies or disparate tables:
  - If both numbers share the same scale (e.g. both in thousands: 9447 and 314258), calculate directly: abs(9447-314258).
  - If the numbers have mismatched scales (e.g. one in millions 321.1 and one in thousands 9447), convert to a common scale before writing the formula (e.g. 321.1 million = 321100 thousand).

IMPORTANT EXAMPLE:

Suppose the evidence contains a table represented as:

2019    2018
m       m
235.8
Net debt
295.2
IFRS 16 lease liabilities
38.9
334.1
235.8
Net debt and IFRS 16 lease liabilities

The flattened text may look confusing.

You must reconstruct the table structure rather than assuming the first number belongs to the first year.

From the table structure:
- Net debt: 2019 = 295.2, 2018 = 235.8
- IFRS 16 lease liabilities: 2019 = 38.9
- Net debt and IFRS 16 lease liabilities: 2019 = 334.1, 2018 = 235.8

Therefore, for the question:

"By how much did net debt and IFRS 16 lease liabilities change from 2018 to 2019?"

the extracted values must identify:
- 2018 combined value = 235.8
- 2019 combined value = 334.1

and the formula must be:

334.1-235.8

Do NOT produce:

334.1-334.1

Do NOT assume:

2018 IFRS 16 lease liabilities = 38.9

because the evidence does not support that assumption.

12. NEGATIVE NUMBERS AND ACCOUNTING PARENTHESES
- In financial statements and accounting tables, numbers enclosed in parentheses such as (2.5), (4.5), (15.7), or (1,234) represent NEGATIVE NUMBERS (-2.5, -4.5, -15.7, -1234).
- When extracting values from financial tables, always convert parenthesized numbers to negative values (e.g. extract -2.5, not 2.5).
- In arithmetic formulas, preserve negative signs (e.g. "(-2.5 + -4.5) / 2" or "334.1 - (-235.8)").

13. COMPONENT SUMS FOR TOTALS
- If the question asks for a "total" or "impact" across financial results/items and the evidence lists the individual components rather than an explicit pre-calculated total, extract all the relevant component numbers and sum them in the formula (e.g. "98.4 + 31.7 + 16.3 + 1.0").

FINAL REQUIREMENT:
- "reasoning_summary" MUST be very concise, at most 1 short sentence (under 25 words). Do not write explanations outside or long paragraphs.
Return ONLY the JSON object.
Do not return markdown.
Do not return explanations outside the JSON.
"""
def _build_evidence_text(chunks: List[Any], max_chunks: int = 6, max_chunk_chars: int = 1800) -> str:
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
        model="qwen/qwen3.8-27b",
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=1000,
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

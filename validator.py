"""
Validator Service

The validation layer that sits between the agent and the user.
Every answer produced by the reasoning pipeline must pass through
here before it is returned.

Validation logic covers:
  - answer_type: must be one of the four allowed types
    (direct, calculated, multi_span, insufficient_evidence)
  - evidence: must be a non-empty list of citations with document_id
    and page for every answer type except insufficient_evidence
  - params: must contain the correct keys and data types for the
    declared answer_type (e.g. 'formula' is required for calculated,
    'values' must be a non-empty list for multi_span)

Returns (True, "") on success, or (False, "<reason>") on failure.
The reason string is logged to the console by main.py in the format:
  [ANSWER-VALIDATOR-SUCCESS] ...
  [ANSWER-VALIDATOR-ERROR]   ...
"""
from typing import Any, Dict, Tuple

VALID_TYPES = {"direct", "calculated", "multi_span", "insufficient_evidence"}

def validate_answer(payload: Dict[str, Any]) -> Tuple[bool, str]:

    answer_type = payload.get("answer_type")

    # 1. Check for answer type
    if not answer_type:
        return False, "Missing required field: 'answer_type'"
    if answer_type not in VALID_TYPES:
        return False, f"Invalid answer_type '{answer_type}'. Must be one of: {VALID_TYPES}"

    evidence = payload.get("evidence")

    # 2. Check for evidence  
    if evidence is None:
        return False, "Missing required field: 'evidence'"
    if not isinstance(evidence, list):
        return False, "'evidence' field must be an array"
    if answer_type != "insufficient_evidence" and len(evidence) == 0:
        return False, "Missing required evidence citation"

    # 3. Check for each citation's info
    for i, cite in enumerate(evidence):
        if not isinstance(cite, dict):
            return False, f"evidence[{i}] must be an object"
        if not cite.get("document_id"):
            return False, f"evidence[{i}] missing 'document_id'"
        if cite.get("page") is None:
            return False, f"evidence[{i}] missing 'page'"

    # 4. Check params field exists
    params = payload.get("params")
    if params is None:
        return False, "Missing required field: 'params'"
    if not isinstance(params, dict):
        return False, "'params' must be an object"

    # 6. Type-specific param checks
    if answer_type == "direct":
        if "value" not in params:
            return False, "Invalid answer for 'direct': Missing required key 'value'"

    elif answer_type == "calculated":
        if "value" not in params:
            return False, "Invalid answer for 'calculated': Missing required key 'value'"
        if "formula" not in params or not params["formula"]:
            return False, "Invalid answer for 'calculated': Missing required key 'formula'"
        if not isinstance(params["value"], (int, float)):
            return False, "Invalid answer for 'calculated': 'value' must be a number"

    elif answer_type == "multi_span":
        if "values" not in params:
            return False, "Invalid answer for 'multi_span': Missing required key 'values'"
        if not isinstance(params["values"], list) or len(params["values"]) == 0:
            return False, "Invalid answer for 'multi_span': 'values' must be a non-empty array"

    elif answer_type == "insufficient_evidence":
        if "reason" not in params or not params["reason"]:
            return False, "Invalid answer for 'insufficient_evidence': Missing required key 'reason'"


    return True, ""
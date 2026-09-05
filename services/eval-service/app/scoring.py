from __future__ import annotations

import math
import re
from typing import Any

from .schemas import Scale

_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def scale_multiplier(scale: Scale) -> float:
    return {
        Scale.NONE: 1.0,
        Scale.THOUSAND: 1_000.0,
        Scale.MILLION: 1_000_000.0,
        Scale.BILLION: 1_000_000_000.0,
    }[scale]


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(",", "")
    return s


def try_parse_number(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip()
    # handle parentheses for negatives: (92) -> -92
    if re.fullmatch(r"\(\s*[-+]?\d[\d,]*\.?\d*\s*\)", s):
        s = "-" + s.strip("()").strip()

    m = _NUM_RE.search(s)
    if not m:
        return None

    num = m.group(0).replace(",", "")
    try:
        return float(num)
    except ValueError:
        return None


def exact_match(pred: Any, gold: Any) -> float:
    if pred is None or gold is None:
        return 0.0
    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        return 1.0 if float(pred) == float(gold) else 0.0
    return 1.0 if normalize_text(str(pred)) == normalize_text(str(gold)) else 0.0


def token_f1(pred: str, gold: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = {}
    for t in p:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in g:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    precision = overlap / len(p)
    recall = overlap / len(g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def numeric_close(pred_num: float, gold_num: float, abs_tol: float = 1e-6, rel_tol: float = 1e-4) -> bool:
    return math.isclose(pred_num, gold_num, abs_tol=abs_tol, rel_tol=rel_tol)


def score_prediction(
    predicted_value: Any,
    gold_value: Any,
    scale: Scale,
) -> tuple[float, float, bool | None]:
    """
    Returns (em, f1, numeric_ok).
    numeric_ok is None when either side is not numeric.
    """
    # Multi-span case
    if isinstance(gold_value, list):
        # very simple list scoring for Phase 1: exact match of normalized joined text
        pred_str = str(predicted_value)
        gold_str = str(gold_value)
        em = exact_match(pred_str, gold_str)
        f1 = token_f1(pred_str, gold_str)
        return em, f1, None

    # Try numeric
    gnum = try_parse_number(gold_value)
    pnum = try_parse_number(predicted_value)
    if gnum is not None and pnum is not None:
        mult = scale_multiplier(scale)
        g = gnum * mult
        p = pnum * mult
        ok = numeric_close(p, g)
        em = 1.0 if ok else 0.0
        return em, em, ok

    # Text scoring
    em = exact_match(predicted_value, gold_value)
    f1 = token_f1(str(predicted_value), str(gold_value))
    return em, f1, None

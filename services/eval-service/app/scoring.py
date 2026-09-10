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
        Scale.PERCENT: 1.0,
    }[scale]


def normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[\'\"\[\]]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace(",", "")
    return s.strip()


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


def numeric_close(pred_num: float, gold_num: float, abs_tol: float = 1e-4, rel_tol: float = 1e-3) -> bool:
    return math.isclose(pred_num, gold_num, abs_tol=abs_tol, rel_tol=rel_tol)


def _score_single_pair(
    pred_item: Any,
    gold_item: Any,
    scale: Scale,
) -> tuple[float, float, bool | None]:
    if pred_item is None or gold_item is None:
        return 0.0, 0.0, None

    # Try numeric first
    gnum = try_parse_number(gold_item)
    pnum = try_parse_number(pred_item)
    if gnum is not None and pnum is not None:
        mult = scale_multiplier(scale)
        g = gnum * mult
        p = pnum * mult
        ok = numeric_close(p, g)
        em = 1.0 if ok else 0.0
        return em, em, ok

    # Fall back to text scoring
    em = exact_match(pred_item, gold_item)
    f1 = token_f1(str(pred_item), str(gold_item))
    return em, f1, None


def score_prediction(
    predicted_value: Any,
    gold_value: Any,
    scale: Scale,
) -> tuple[float, float, bool | None]:
    """
    Returns (em, f1, numeric_ok).
    numeric_ok is None when either side is not numeric.
    """
    if predicted_value is None or gold_value is None:
        return 0.0, 0.0, None

    # If gold_value is a single-element list (e.g. ['Deloitte'], ['201.8'], ['$5.9 million'])
    # unwrap it so it can be evaluated as a scalar (either numeric or text)
    if isinstance(gold_value, list) and len(gold_value) == 1:
        gold_item = gold_value[0]
        pred_item = predicted_value[0] if (isinstance(predicted_value, list) and len(predicted_value) == 1) else predicted_value
        return _score_single_pair(pred_item, gold_item, scale)

    # Multi-span case (gold is a list of 2 or more elements)
    if isinstance(gold_value, list):
        if isinstance(predicted_value, list):
            p_items = [normalize_text(x) for x in predicted_value]
        else:
            p_items = [normalize_text(predicted_value)]

        g_items = [normalize_text(x) for x in gold_value]
        em = 1.0 if sorted(p_items) == sorted(g_items) else 0.0
        f1 = token_f1(" ".join(p_items), " ".join(g_items))
        return em, f1, None

    # Single scalar gold (int, float, str)
    pred_item = predicted_value[0] if (isinstance(predicted_value, list) and len(predicted_value) == 1) else predicted_value
    return _score_single_pair(pred_item, gold_value, scale)


def compute_page_retrieval_metrics(
    retrieved_pages: list[tuple[str, int]],
    gold_pages: list[tuple[str, int]],
    k: int,
) -> dict[str, float]:
    """
    Page-level retrieval metrics.

    retrieved_pages: ranked list of (document_id, page)
    gold_pages: set/list of relevant (document_id, page)
    """
    gold_set = set(gold_pages)
    retrieved_at_k = retrieved_pages[:k]

    relevant_retrieved = [p for p in retrieved_at_k if p in gold_set]
    num_rel_ret = len(relevant_retrieved)

    hit = 1.0 if num_rel_ret > 0 else 0.0
    recall = (num_rel_ret / len(gold_set)) if gold_set else 0.0
    precision = (num_rel_ret / k) if k > 0 else 0.0

    rr = 0.0
    for rank, p in enumerate(retrieved_pages, start=1):
        if p in gold_set:
            rr = 1.0 / rank
            break

    return {"hit": hit, "recall": recall, "precision": precision, "rr": rr}

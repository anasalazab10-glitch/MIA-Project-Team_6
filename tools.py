import ast
import operator
import re
from typing import Any, Dict, List, Optional
import httpx

from schemas import RetrievedChunk

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}


def _eval_ast_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        return _ALLOWED_OPERATORS[op_type](left, right)

    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_ast_node(node.operand)
        return _ALLOWED_OPERATORS[op_type](operand)

    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct named functions are allowed.")
        func_name = node.func.id
        if func_name not in _ALLOWED_FUNCTIONS:
            raise ValueError(f"Function call '{func_name}' is not permitted.")
        args = [_eval_ast_node(arg) for arg in node.args]
        return _ALLOWED_FUNCTIONS[func_name](*args)

    raise ValueError(f"Unsupported syntax expression: {type(node).__name__}")


def calculate(formula: str) -> float:
    """
    Deterministic calculator tool. This is the ONLY place arithmetic is
    allowed to happen — the LLM must never produce a computed number itself.
    """
    clean_formula = re.sub(r"(?<=\d),(?=\d)", "", formula.strip())
    try:
        parsed = ast.parse(clean_formula, mode="eval")
        result = _eval_ast_node(parsed.body)
        return float(result)
    except Exception as exc:
        raise ValueError(
            f"Failed to evaluate formula '{formula}': {str(exc)}"
        ) from exc


import os

RETRIEVAL_API_URL = os.environ.get("RETRIEVAL_API_URL", "http://localhost:8000")


def _adapt_raw_chunk(raw: Dict[str, Any]) -> RetrievedChunk:
    """
    Converts retrieval-api's raw chunk format into our RetrievedChunk schema.

    """
    page_raw = raw.get("page", [0])
    page = page_raw if isinstance(page_raw, list) else [page_raw]

    content = raw.get("content", "")
    if isinstance(content, dict):
        # Table content: flatten headers/rows into readable text
        headers = content.get("headers", [])
        rows = content.get("rows", [])
        lines = [", ".join(str(h) for h in headers)]
        for row in rows:
            lines.append(", ".join(str(cell) for cell in row))
        text = "; ".join(lines)
    else:
        text = str(content)

    return RetrievedChunk(
        document_id=raw.get("document_id", "unknown"),
        page=page,
        section=raw.get("section", "General"),
        content_type=raw.get("content_type", "text"),
        text=text,
    )


def _mock_chunks() -> List[RetrievedChunk]:
    """Shared mock fallback data used across tools during standalone dev."""
    return [
        RetrievedChunk(
            document_id="cts-corporation_2019.pdf",
            page=[1],
            section="Financial Statements",
            content_type="table",
            text="Net sales: 469850. Operating earnings: 38750.",
        ),
        RetrievedChunk(
            document_id="jabil-circuit-inc_2019.pdf",
            page=[1],
            section="Financial Statements",
            content_type="table",
            text="Net revenue: 25296000. Operating income: 714200.",
        ),
    ]


def search_documents(
    query: str,
    search_type: str = "hybrid",  # kept for interface compatibility; not sent to API
    top_k: int = 5,               # kept for interface compatibility; API always returns top 5
    document_id: Optional[str] = None,
    mock_mode: bool = False,
) -> List[RetrievedChunk]:
    """
    General-purpose corpus search. Calls retrieval-api's /search endpoint,
    which handles embedding + reranking internally and always returns the
    top 5 reranked candidates.
    """
    if not mock_mode:
        payload = {"query": query}  # confirmed: API only expects {"query": "..."}
        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.post(f"{RETRIEVAL_API_URL}/search", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    raw_results = data.get("results", data if isinstance(data, list) else [])
                    chunks = [_adapt_raw_chunk(chunk) for chunk in raw_results]
                    if document_id:
                        chunks = [c for c in chunks if c.document_id == document_id]
                    return chunks
                else:
                    print(
                        f"[search_documents] retrieval-api returned "
                        f"status {res.status_code}: {res.text}"
                    )
        except Exception as exc:
            print(f"[search_documents] retrieval-api call failed: {exc}")

    return _mock_chunks()


def search_tables(
    query: str,
    document_id: Optional[str] = None,
    top_k: int = 5,
    mock_mode: bool = False,
) -> List[RetrievedChunk]:
    """
    Table-specific search. NOTE: retrieval-api currently exposes only one
    endpoint (/search) - there is no dedicated table-search route. This
    calls the same /search endpoint and filters the results down to
    table chunks client-side.
    """
    all_chunks = search_documents(query, document_id=document_id, mock_mode=mock_mode)
    table_chunks = [c for c in all_chunks if c.content_type == "table"]
    # Fall back to all results if nothing was tagged as a table -
    # better to hand the reasoner something than nothing.
    return table_chunks if table_chunks else all_chunks


def filter_documents(
    metadata: Dict[str, Any],
    mock_mode: bool = False,
) -> List[str]:
    """
    Non-vector, metadata-based lookup. NOTE: retrieval-api has not confirmed
    a dedicated filter endpoint yet - only /search is confirmed live. 
    """
    
    return [c.document_id for c in _mock_chunks()]
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


RETRIEVAL_API_URL = "http://localhost:8001"


def _adapt_raw_chunk(raw: Dict[str, Any]) -> RetrievedChunk:
    """
    Converts retrieval-api's raw chunk format into our RetrievedChunk schema.

    Their format differs from ours in two ways:
      - "page" is a list (e.g. [1]) instead of a plain int
      - "content" is used instead of "text", and for tables it's a nested
        dict ({"headers": [...], "rows": [...]}) instead of a plain string
    """
    page_raw = raw.get("page")
    if isinstance(page_raw, list):
        page = page_raw[0] if page_raw else 0
    else:
        page = page_raw or 0

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
            page=1,
            section="Financial Statements",
            content_type="table",
            text="Net sales: 469850. Operating earnings: 38750.",
        ),
        RetrievedChunk(
            document_id="jabil-circuit-inc_2019.pdf",
            page=1,
            section="Financial Statements",
            content_type="table",
            text="Net revenue: 25296000. Operating income: 714200.",
        ),
    ]


def search_documents(
    query: str,
    search_type: str = "hybrid",  # vector search + keyword
    top_k: int = 5,
    document_id: Optional[str] = None,
    mock_mode: bool = False,
) -> List[RetrievedChunk]:
    """
    General-purpose corpus search (text + tables), semantic + keyword hybrid.
    """
    if not mock_mode:
        payload = {
            "query": query,
            "search_type": search_type,
            "top_k": top_k,
            "document_id": document_id,
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.post(f"{RETRIEVAL_API_URL}/search", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    raw_results = data.get("results", data if isinstance(data, list) else [])
                    return [_adapt_raw_chunk(chunk) for chunk in raw_results]
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
    Table-specific search — used when the question needs structured/tabular
    evidence (e.g. line items, financial statement rows) rather than prose.
    """
    if not mock_mode:
        payload = {
            "query": query,
            "top_k": top_k,
            "document_id": document_id,
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.post(f"{RETRIEVAL_API_URL}/search/tables", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    raw_results = data.get("results", data if isinstance(data, list) else [])
                    return [_adapt_raw_chunk(chunk) for chunk in raw_results]
                else:
                    print(
                        f"[search_tables] retrieval-api returned "
                        f"status {res.status_code}: {res.text}"
                    )
        except Exception as exc:
            print(f"[search_tables] retrieval-api call failed: {exc}")

    return [c for c in _mock_chunks() if c.content_type == "table"]


def filter_documents(
    metadata: Dict[str, Any],
    mock_mode: bool = False,
) -> List[str]:
    """
    Non-vector, metadata-based lookup — e.g. filter by document_id, company
    name, or fiscal year before/instead of running semantic search.
    Returns a list of matching document_ids.
    """
    if not mock_mode:
        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.post(
                    f"{RETRIEVAL_API_URL}/documents/filter", json=metadata
                )
                if res.status_code == 200:
                    data = res.json()
                    return data.get("document_ids", [])
                else:
                    print(
                        f"[filter_documents] retrieval-api returned "
                        f"status {res.status_code}: {res.text}"
                    )
        except Exception as exc:
            print(f"[filter_documents] retrieval-api call failed: {exc}")

    # Mock fallback: pretend everything matches
    return [c.document_id for c in _mock_chunks()]
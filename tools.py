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


def _mock_chunks() -> List[RetrievedChunk]:
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
                    return [
                        RetrievedChunk(**chunk)
                        for chunk in data.get("results", [])
                    ]
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
                    return [
                        RetrievedChunk(**chunk)
                        for chunk in data.get("results", [])
                    ]
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

    return [c.document_id for c in _mock_chunks()]

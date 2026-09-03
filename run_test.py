"""
run_test.py

Offline benchmark harness: loads TAT-DQA held-out questions, runs each one
through the compiled agent graph, and scores predicted answers against
ground truth (Exact Match, F1, numerical accuracy).

STATUS: graph.py isn't built yet, so `run_agent_graph()` below is a
PLACEHOLDER. Everything else - loading questions, scoring, reporting - is
real and can be tested now with fake/mock predictions.

Once graph.py exists, replace ONLY the body of `run_agent_graph()` with:
    from graph import compiled_graph
    result_state = compiled_graph.invoke({"question": question, "session_id": None})
    return result_state["final_answer"]
"""
import json
from pathlib import Path
from typing import Any, Dict, List


QUESTIONS_FILE = "questions_setA_practice.json"
ANSWER_TYPE_MAP = {
    "arithmetic": "calculated",
    "span": "direct",
    "multi-span": "multi_span",
    "unanswerable": "insufficient_evidence",
    "count": "direct",
}

# PLACEHOLDER
def run_agent_graph(question: str) -> Dict[str, Any]:
    """
    TEMPORARY stand-in for the real compiled LangGraph. Returns a fake
    'direct' answer so the rest of the harness can be built/tested now.
    """
    return {
        "answer_type": "direct",
        "evidence": [{"document_id": "mock_doc", "page": 1, "section": "General"}],
        "params": {"value": "MOCK_ANSWER"},
    }
# ----------------------------------------------


def load_questions(path: str) -> List[Dict[str, Any]]:
    """
    Loads the held-out question set. Expected format :
        [
          {"question": "...", "answer": "...", "answer_type": "direct", ...},
          ...
        ]
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Couldn't find {path}. Place the practice question set in the "
            f"same folder as this script, or update QUESTIONS_FILE."
        )
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_value(value: Any) -> str:
    """Lowercase + strip, so '142.5M' and '142.5m ' compare equal, etc."""
    return str(value).strip().lower()


def exact_match(predicted: Any, ground_truth: Any) -> bool:
    return normalize_value(predicted) == normalize_value(ground_truth)


def f1_score(predicted: str, ground_truth: str) -> float:
    """Simple token-overlap F1, standard for QA benchmarks."""
    pred_tokens = normalize_value(predicted).split()
    gt_tokens = normalize_value(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def numerical_accuracy(predicted: Any, ground_truth: Any, tolerance: float = 0.01) -> bool:
    """For 'calculated' answers - allow small floating point tolerance."""
    try:
        return abs(float(predicted) - float(ground_truth)) <= tolerance
    except (TypeError, ValueError):
        return False


def score_one(question_item: Dict[str, Any], predicted_answer: Dict[str, Any]) -> Dict[str, Any]:
    """Compares one predicted answer against its ground truth."""
    gt = question_item.get("ground_truth_answer")
    predicted_params = predicted_answer.get("params", {})
    predicted_value = predicted_params.get("value") or predicted_params.get("values")

    result = {
        "question": question_item.get("question_text"),
        "predicted": predicted_value,
        "ground_truth": gt,
        "answer_type": predicted_answer.get("answer_type"),
        "exact_match": exact_match(predicted_value, gt),
        "f1": f1_score(str(predicted_value), str(gt)),
    }

    if predicted_answer.get("answer_type") == "calculated":
        result["numerical_correct"] = numerical_accuracy(predicted_value, gt)

    return result


def run_benchmark(questions: List[Dict[str, Any]]) -> None:
    all_results = []

    for item in questions:
        question_text = item["question_text"]
        predicted = run_agent_graph(question_text)
        all_results.append(score_one(item, predicted))

    total = len(all_results)
    em_count = sum(r["exact_match"] for r in all_results)
    avg_f1 = sum(r["f1"] for r in all_results) / total if total else 0

    print(f"\n===== Benchmark Results ({total} questions) =====")
    print(f"Exact Match: {em_count}/{total} ({100 * em_count / total:.1f}%)")
    print(f"Average F1:  {avg_f1:.3f}")

    print("\n--- Sample results ---")
    for r in all_results[:5]:
        print(r)


if __name__ == "__main__":
    questions = load_questions(QUESTIONS_FILE)
    run_benchmark(questions)
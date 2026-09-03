"""
run_test.py

Runs the complete agent pipeline (graph.py) against a set of real practice
questions and prints what each one produced
"""
import json
from typing import Any, Dict, List

from dotenv import load_dotenv
load_dotenv()

from graph import agent_graph

QUESTIONS_FILE = "questions_setA_practice.json"
MAX_QUESTIONS = 10  # start small; set to None to run the full set


def load_questions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list, got {type(data)}")
    return data


def run_pipeline():
    questions = load_questions(QUESTIONS_FILE)
    if MAX_QUESTIONS:
        questions = questions[:MAX_QUESTIONS]

    total = len(questions)
    insufficient_count = 0
    errors = []

    print(f"Running {total} questions from {QUESTIONS_FILE} through the pipeline...\n")

    for i, item in enumerate(questions, 1):
        question_text = item["question_text"]

        print(f"{'=' * 70}")
        print(f"[{i}/{total}] {question_text}")
        print("-" * 70)

        try:
            initial_state = {
                "question": question_text,
                "session_id": f"run-{item.get('question_id', i)}",
                "retrieved_chunks": [],
            }
            result_state = agent_graph.invoke(initial_state)
            final_answer = result_state.get("final_answer", {})
        except Exception as exc:
            print(f"ERROR running pipeline: {exc}")
            errors.append({"question": question_text, "error": str(exc)})
            continue

        if final_answer.get("answer_type") == "insufficient_evidence":
            insufficient_count += 1

        print(f"answer_type     : {final_answer.get('answer_type')}")
        print(f"params          : {final_answer.get('params')}")
        print(f"evidence        : {final_answer.get('evidence')}")
        print(f"retry_count     : {result_state.get('retry_count')}")
        print()

    print("=" * 70)
    print("RUN SUMMARY")
    print("=" * 70)
    print(f"Total questions run        : {total}")
    print(f"Returned insufficient_evidence: {insufficient_count}/{total}")
    print(f"Pipeline errors             : {len(errors)}")

    if errors:
        print("\nFirst few errors:")
        for e in errors[:5]:
            print(f"  - {e['question'][:60]}...: {e['error']}")


if __name__ == "__main__":
    run_pipeline()
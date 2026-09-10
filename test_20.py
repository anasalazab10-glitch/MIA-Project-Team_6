
"""
test_20.py - Runs the first 20 questions from the benchmark results file
against the LIVE orchestrator and reports pass/fail vs ground truth.
"""

import json
import sys
import requests


import os
import time

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8003/run")
BENCHMARK_FILE = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("BENCHMARK_FILE", "benchmark_100_raw.json")
NUM_QUESTIONS = int(os.environ.get("NUM_QUESTIONS", 10))


def normalize(val):
    if val is None:
        return None

    if isinstance(val, list):
        return [normalize(v) for v in val]

    s = str(val).strip().lower()
    s = s.replace("per cent", "%").replace(" %", "%")
    s = s.rstrip("%").strip()
    s = s.replace(",", "").replace("$", "").replace("£", "")

    try:
        return round(float(s), 2)
    except ValueError:
        return s.rstrip(".").strip()


def extract_predicted_value(resp_json):
    params = resp_json.get("params", {})

    if "value" in params:
        return params["value"]

    if "values" in params:
        return params["values"]

    return None


def extract_reason(resp_json):
    return resp_json.get("params", {}).get("reason")


def main():
    with open(BENCHMARK_FILE, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        questions = data[:NUM_QUESTIONS]
    elif "results" in data and isinstance(data["results"], dict) and "results" in data["results"]:
        questions = data["results"]["results"][:NUM_QUESTIONS]
    elif "results" in data and isinstance(data["results"], list):
        questions = data["results"][:NUM_QUESTIONS]
    else:
        questions = []

    results = []

    for i, q in enumerate(questions, 1):
        qid = q.get("question_id", f"Q{i}")
        qtext = q["question_text"]
        gt = q.get("ground_truth_answer")

        print(f"\n[{i}/{NUM_QUESTIONS}] {qid}: {qtext}")

        try:
            resp = requests.post(
                ORCHESTRATOR_URL,
                json={"question": qtext},
                timeout=90,
            )

            resp_json = resp.json()

        except Exception as exc:
            print(f"  ERROR: {exc}")

            results.append(
                (qid, qtext, gt, None, "ERROR")
            )

            continue

        answer_type = resp_json.get("answer_type")
        predicted = extract_predicted_value(resp_json)

        gt_norm = normalize(gt)
        pred_norm = normalize(predicted)

        if isinstance(gt_norm, list) and len(gt_norm) == 1:
            gt_norm = gt_norm[0]
        if isinstance(pred_norm, list) and len(pred_norm) == 1:
            pred_norm = pred_norm[0]

        is_unanswerable_gt = gt is None
        got_insufficient = answer_type == "insufficient_evidence"

        if is_unanswerable_gt and got_insufficient:
            status = "CORRECT (unanswerable)"

        elif is_unanswerable_gt and not got_insufficient:
            status = "INCORRECT (should be unanswerable)"

        elif not is_unanswerable_gt and got_insufficient:
            status = "INCORRECT (wrongly insufficient)"

        elif isinstance(gt_norm, list) and isinstance(pred_norm, list):
            overlap = len(set(gt_norm) & set(pred_norm))

            if overlap == len(gt_norm):
                status = "CORRECT"
            else:
                status = f"PARTIAL ({overlap}/{len(gt_norm)})"

        elif gt_norm == pred_norm:
            status = "CORRECT"

        else:
            status = "INCORRECT"

        print(f"  Ground truth: {gt}")
        print(f"  Predicted:    {predicted}  (type: {answer_type})")

        if answer_type == "insufficient_evidence":
            print(f"  Reason:       {extract_reason(resp_json)}")

        print(f"  Status:       {status}")

        # Store the result so the final summary works
        results.append(
            (qid, qtext, gt, predicted, status)
        )
        time.sleep(2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    correct = sum(
        1 for r in results
        if r[4].startswith("CORRECT")
    )

    print(f"Correct: {correct}/{len(results)}")

    for qid, qtext, gt, pred, status in results:
        marker = "✅" if status.startswith("CORRECT") else "❌"
        print(f"{marker} {qid}: {status}")


if __name__ == "__main__":
    main()


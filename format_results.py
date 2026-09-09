import json
from pathlib import Path


INPUT_FILE = "benchmark_100_raw.json"
OUTPUT_FILE = "benchmark_100_report.md"


def format_value(value):
    """Format values nicely for the Markdown report."""
    if value is None:
        return "N/A"

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def main():
    input_path = Path(INPUT_FILE)

    if not input_path.exists():
        print(f"ERROR: {INPUT_FILE} was not found.")
        print("Run run_100_questions.py first.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", data)

    # ---------------------------------------------------------
    # Extract benchmark response
    # ---------------------------------------------------------

    if isinstance(results, dict):
        questions = results.get("results", [])

        # Overall metrics returned directly by Eval Service
        num_items = results.get("num_items")
        num_scored = results.get("num_scored")

        avg_em = results.get("avg_em")
        avg_f1 = results.get("avg_f1")
        numeric_accuracy = results.get("numeric_accuracy")

        avg_hit_at_k = results.get("avg_hit_at_k")
        avg_recall_at_k = results.get("avg_recall_at_k")
        avg_precision_at_k = results.get("avg_precision_at_k")
        mrr_at_k = results.get("mrr_at_k")

    elif isinstance(results, list):
        questions = results

        num_items = None
        num_scored = None
        avg_em = None
        avg_f1 = None
        numeric_accuracy = None
        avg_hit_at_k = None
        avg_recall_at_k = None
        avg_precision_at_k = None
        mrr_at_k = None

    else:
        questions = []

        num_items = None
        num_scored = None
        avg_em = None
        avg_f1 = None
        numeric_accuracy = None
        avg_hit_at_k = None
        avg_recall_at_k = None
        avg_precision_at_k = None
        mrr_at_k = None

    # ---------------------------------------------------------
    # Build report
    # ---------------------------------------------------------

    report = []

    report.append("# LEDGER — 100 Question Benchmark Report\n")

    # ---------------------------------------------------------
    # Run Information
    # ---------------------------------------------------------

    report.append("## 1. Run Information\n")

    started_at = data.get("started_at", "N/A")
    finished_at = data.get("finished_at", "N/A")
    duration = data.get("duration_seconds", "N/A")

    report.append(f"- **Started:** {started_at}")
    report.append(f"- **Finished:** {finished_at}")
    report.append(f"- **Duration:** {format_value(duration)} seconds")
    report.append(f"- **Number of questions:** {num_items or len(questions)}")
    report.append(f"- **Number scored:** {num_scored or len(questions)}\n")

    # ---------------------------------------------------------
    # Overall Evaluation Metrics
    # ---------------------------------------------------------

    report.append("## 2. Overall Evaluation Metrics\n")

    report.append("| Metric | Score |")
    report.append("|---|---:|")

    report.append(f"| Average EM | {format_value(avg_em)} |")
    report.append(f"| Average F1 | {format_value(avg_f1)} |")
    report.append(
        f"| Numeric Accuracy | {format_value(numeric_accuracy)} |"
    )
    report.append(
        f"| Average Hit@K | {format_value(avg_hit_at_k)} |"
    )
    report.append(
        f"| Average Recall@K | {format_value(avg_recall_at_k)} |"
    )
    report.append(
        f"| Average Precision@K | {format_value(avg_precision_at_k)} |"
    )
    report.append(
        f"| MRR@K | {format_value(mrr_at_k)} |"
    )

    report.append("")

    # ---------------------------------------------------------
    # Individual Results
    # ---------------------------------------------------------

    report.append("## 3. Question-by-Question Results\n")

    for index, item in enumerate(questions, start=1):

        question_id = (
            item.get("question_id")
            or item.get("id")
            or f"Question {index}"
        )

        question = (
            item.get("question_text")
            or item.get("question")
            or item.get("query")
            or "N/A"
        )

        # Ground truth
        expected = (
            item.get("ground_truth_answer")
            if item.get("ground_truth_answer") is not None
            else item.get("expected")
        )

        if expected is None:
            expected = (
                item.get("ground_truth")
                or item.get("reference_answer")
                or "N/A"
            )

        # Prediction
        predicted_answer = item.get("predicted_answer")

        if isinstance(predicted_answer, dict):
            predicted = predicted_answer.get(
                "params", {}
            ).get("value")

            if predicted is None:
                predicted = predicted_answer.get(
                    "answer",
                    predicted_answer.get("value", "N/A")
                )

            answer_type = predicted_answer.get(
                "answer_type", "N/A"
            )

        else:
            predicted = (
                item.get("predicted")
                or item.get("prediction")
                or item.get("final_answer")
                or "N/A"
            )

            answer_type = item.get("answer_type", "N/A")

        # Metrics
        em = item.get("em")
        f1 = item.get("f1")
        numeric_ok = item.get("numeric_ok")

        retrieval_hit = item.get("retrieval_hit")
        retrieval_recall = item.get("retrieval_recall")
        retrieval_precision = item.get("retrieval_precision")
        retrieval_rr = item.get("retrieval_rr")

        validation_status = item.get(
            "validation_status",
            "N/A"
        )

        validation_reason = item.get(
            "validation_reason"
        )

        error = item.get("error")

        trace_id = item.get("trace_id")

        # -----------------------------------------------------
        # Question heading
        # -----------------------------------------------------

        report.append(
            f"### {index}. {question_id}\n"
        )

        report.append(
            f"**Question:** {question}\n"
        )

        report.append(
            f"**Ground Truth:** `{expected}`  \n"
        )

        report.append(
            f"**Predicted Answer:** `{predicted}`  \n"
        )

        report.append(
            f"**Answer Type:** `{answer_type}`  \n"
        )

        # -----------------------------------------------------
        # Evaluation metrics
        # -----------------------------------------------------

        report.append("#### Evaluation Metrics\n")

        report.append("| Metric | Score |")
        report.append("|---|---:|")

        report.append(
            f"| Exact Match (EM) | {format_value(em)} |"
        )

        report.append(
            f"| F1 | {format_value(f1)} |"
        )

        report.append(
            f"| Numeric Correct | {format_value(numeric_ok)} |"
        )

        report.append(
            f"| Retrieval Hit@K | {format_value(retrieval_hit)} |"
        )

        report.append(
            f"| Retrieval Recall@K | "
            f"{format_value(retrieval_recall)} |"
        )

        report.append(
            f"| Retrieval Precision@K | "
            f"{format_value(retrieval_precision)} |"
        )

        report.append(
            f"| Retrieval RR | {format_value(retrieval_rr)} |"
        )

        report.append("")

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        report.append("#### Validation\n")

        report.append(
            f"- **Validation Status:** `{validation_status}`"
        )

        if validation_reason:
            report.append(
                f"- **Validation Reason:** {validation_reason}"
            )

        report.append("")

        # -----------------------------------------------------
        # Langfuse
        # -----------------------------------------------------

        report.append("#### Langfuse\n")

        if trace_id:
            report.append(
                f"- **Trace ID:** `{trace_id}`"
            )
        else:
            report.append(
                "- **Trace ID:** Not included in Eval Service response "
                "(trace is still handled by Langfuse)"
            )

        report.append("")

        # -----------------------------------------------------
        # Error
        # -----------------------------------------------------

        if error:
            report.append("#### Error\n")
            report.append(f"`{error}`\n")

        # -----------------------------------------------------
        # Status
        # -----------------------------------------------------

        if em == 1:
            status = "✅ CORRECT"
        elif numeric_ok is True:
            status = "✅ NUMERICALLY CORRECT"
        else:
            status = "❌ INCORRECT"

        report.append(f"**Status:** {status}\n")

        report.append("---\n")

    # ---------------------------------------------------------
    # Write report
    # ---------------------------------------------------------

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("=" * 60)
    print("Formatted report created.")
    print("=" * 60)
    print(f"File: {OUTPUT_FILE}")
    print(f"Questions found: {len(questions)}")


if __name__ == "__main__":
    main()
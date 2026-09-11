
import json
import os
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

RESULTS_FILE = Path(
    "services/eval-service/results/benchmark_results.json"
)

OUTPUT_FILE = Path(
    "services/eval-service/results/benchmark_report.md"
)


# ============================================================
# Helpers
# ============================================================

def format_value(value):
    """Format values for readable output."""

    if value is None:
        return "N/A"

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def shorten(text, max_length=120):
    """Shorten long text for tables."""

    if text is None:
        return "N/A"

    text = str(text).replace("\n", " ").strip()

    if len(text) <= max_length:
        return text

    return text[:max_length - 3] + "..."


def extract_prediction(result):
    """
    Extract the actual predicted value from the
    orchestrator response.
    """

    predicted = result.get("predicted_answer")

    if not isinstance(predicted, dict):
        return "N/A"

    answer_type = predicted.get(
        "answer_type",
        "unknown",
    )

    params = predicted.get(
        "params"
    ) or {}

    if answer_type in (
        "direct",
        "calculated",
    ):
        return params.get(
            "value",
            "N/A",
        )

    if answer_type == "multi_span":
        values = params.get(
            "values",
            []
        )

        if isinstance(values, list):
            return "; ".join(
                str(value)
                for value in values
            )

        return values

    if answer_type == "insufficient_evidence":
        return "INSUFFICIENT EVIDENCE"

    return "N/A"


def extract_answer_type(result):
    predicted = result.get(
        "predicted_answer"
    )

    if isinstance(predicted, dict):
        return predicted.get(
            "answer_type",
            "N/A",
        )

    return "N/A"


def count_answer_types(results):
    counts = {}

    for result in results:

        answer_type = extract_answer_type(
            result
        )

        counts[answer_type] = (
            counts.get(answer_type, 0) + 1
        )

    return counts


def count_errors(results):
    return sum(
        1
        for result in results
        if result.get("error")
    )


# ============================================================
# Generate report
# ============================================================

def generate_report():

    if not RESULTS_FILE.exists():

        raise FileNotFoundError(
            f"Results file not found: "
            f"{RESULTS_FILE}"
        )

    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    results = data.get(
        "results",
        []
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    num_items = data.get(
        "num_items",
        len(results),
    )

    num_scored = data.get(
        "num_scored",
        0,
    )

    avg_em = data.get(
        "avg_em"
    )

    avg_f1 = data.get(
        "avg_f1"
    )

    numeric_accuracy = data.get(
        "numeric_accuracy"
    )

    avg_hit = data.get(
        "avg_hit_at_k"
    )

    avg_recall = data.get(
        "avg_recall_at_k"
    )

    avg_precision = data.get(
        "avg_precision_at_k"
    )

    mrr = data.get(
        "mrr_at_k"
    )

    errors = count_errors(
        results
    )

    answer_types = count_answer_types(
        results
    )

    # --------------------------------------------------------
    # Build report
    # --------------------------------------------------------

    report = []

    report.append(
        "# LEDGER Benchmark Evaluation Report\n"
    )

    report.append(
        "## 1. Benchmark Overview\n"
    )

    report.append(
        f"- **Questions evaluated:** {num_items}\n"
        f"- **Questions scored:** {num_scored}\n"
        f"- **Questions with errors:** {errors}\n"
    )

    report.append(
        "\n## 2. Overall Performance\n"
    )

    report.append(
        "| Metric | Result |\n"
        "|---|---:|\n"
    )

    report.append(
        f"| Exact Match (EM) | "
        f"{format_value(avg_em)} |\n"
    )

    report.append(
        f"| F1 Score | "
        f"{format_value(avg_f1)} |\n"
    )

    report.append(
        f"| Numeric Accuracy | "
        f"{format_value(numeric_accuracy)} |\n"
    )

    report.append(
        f"| Hit@5 | "
        f"{format_value(avg_hit)} |\n"
    )

    report.append(
        f"| Recall@5 | "
        f"{format_value(avg_recall)} |\n"
    )

    report.append(
        f"| Precision@5 | "
        f"{format_value(avg_precision)} |\n"
    )

    report.append(
        f"| MRR@5 | "
        f"{format_value(mrr)} |\n"
    )

    # --------------------------------------------------------
    # Answer type distribution
    # --------------------------------------------------------

    report.append(
        "\n## 3. Answer Type Distribution\n"
    )

    report.append(
        "| Answer Type | Number of Questions |\n"
        "|---|---:|\n"
    )

    for answer_type, count in sorted(
        answer_types.items()
    ):

        report.append(
            f"| {answer_type} | {count} |\n"
        )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    report.append(
        "\n## 4. Performance Summary\n"
    )

    if avg_hit is not None and avg_hit >= 0.9:

        report.append(
            "The retrieval component performed strongly "
            "on this benchmark subset. The high Hit@5 "
            "indicates that the relevant evidence was "
            "generally present among the top five "
            "retrieved candidates.\n\n"
        )

    if avg_recall is not None and avg_recall >= 0.9:

        report.append(
            "Recall@5 was high, indicating that the "
            "retrieval pipeline successfully recovered "
            "the gold evidence pages for the evaluated "
            "questions.\n\n"
        )

    if avg_precision is not None and avg_precision < 0.5:

        report.append(
            "Precision@5 was comparatively lower. "
            "This indicates that although the relevant "
            "evidence was retrieved, the top five "
            "results often contained additional "
            "non-relevant candidates.\n\n"
        )

    if avg_em is not None and avg_em < 0.5:

        report.append(
            "End-to-end Exact Match remains relatively "
            "low. This suggests that the main remaining "
            "errors are not necessarily caused by "
            "retrieval alone and may occur during "
            "reasoning, calculation, answer extraction, "
            "or formatting.\n\n"
        )

    if numeric_accuracy is not None:

        report.append(
            f"Numeric accuracy was "
            f"{numeric_accuracy:.2%}, indicating that "
            "a portion of the numerical questions were "
            "answered correctly, while some numerical "
            "reasoning or formatting cases still failed.\n\n"
        )

    # --------------------------------------------------------
    # Detailed results
    # --------------------------------------------------------

    report.append(
        "## 5. Question-by-Question Results\n"
    )

    for index, result in enumerate(
        results,
        start=1,
    ):

        question_id = result.get(
            "question_id",
            "N/A",
        )

        question = result.get(
            "question_text",
            "N/A",
        )

        ground_truth = result.get(
            "ground_truth_answer",
            "N/A",
        )

        prediction = extract_prediction(
            result
        )

        answer_type = extract_answer_type(
            result
        )

        em = result.get(
            "em"
        )

        f1 = result.get(
            "f1"
        )

        numeric_ok = result.get(
            "numeric_ok"
        )

        hit = result.get(
            "retrieval_hit"
        )

        recall = result.get(
            "retrieval_recall"
        )

        precision = result.get(
            "retrieval_precision"
        )

        rr = result.get(
            "retrieval_rr"
        )

        error = result.get(
            "error"
        )

        report.append(
            f"### {index}. {question_id}\n\n"
        )

        report.append(
            f"**Question:**  \n"
            f"{question}\n\n"
        )

        report.append(
            f"**Ground truth:**  \n"
            f"{ground_truth}\n\n"
        )

        report.append(
            f"**Prediction:**  \n"
            f"{prediction}\n\n"
        )

        report.append(
            f"**Answer type:** "
            f"`{answer_type}`\n\n"
        )

        report.append(
            "| Metric | Result |\n"
            "|---|---:|\n"
        )

        report.append(
            f"| EM | {format_value(em)} |\n"
        )

        report.append(
            f"| F1 | {format_value(f1)} |\n"
        )

        report.append(
            f"| Numeric Correct | "
            f"{format_value(numeric_ok)} |\n"
        )

        report.append(
            f"| Hit@5 | {format_value(hit)} |\n"
        )

        report.append(
            f"| Recall@5 | "
            f"{format_value(recall)} |\n"
        )

        report.append(
            f"| Precision@5 | "
            f"{format_value(precision)} |\n"
        )

        report.append(
            f"| RR | {format_value(rr)} |\n"
        )

        if error:

            report.append(
                "\n**Error:**  \n"
                f"`{error}`\n\n"
            )

        # ----------------------------------------------------
        # Highlight failed answers
        # ----------------------------------------------------

        if (
            em is not None
            and em < 1.0
            and not error
        ):

            report.append(
                "\n**Observation:** "
                "The question was processed, but "
                "the predicted answer did not exactly "
                "match the ground truth.\n\n"
            )

        if (
            answer_type
            == "insufficient_evidence"
            and result.get("is_answerable") is True
        ):

            report.append(
                "**Potential issue:** "
                "The benchmark considers this question "
                "answerable, but the system returned "
                "`insufficient_evidence`. This should "
                "be investigated as a possible reasoning "
                "or formatting failure.\n\n"
            )

        report.append(
            "---\n\n"
        )

    # --------------------------------------------------------
    # Failed questions summary
    # --------------------------------------------------------

    failed_results = [
        result
        for result in results
        if (
            result.get("em") is not None
            and result.get("em") < 1.0
        )
    ]

    report.append(
        "## 6. Questions Requiring Investigation\n"
    )

    if not failed_results:

        report.append(
            "All evaluated questions achieved "
            "Exact Match.\n"
        )

    else:

        report.append(
            "| Question | Answer Type | EM | F1 | Issue |\n"
            "|---|---|---:|---:|---|\n"
        )

        for result in failed_results:

            question_id = result.get(
                "question_id",
                "N/A",
            )

            answer_type = extract_answer_type(
                result
            )

            em = result.get(
                "em"
            )

            f1 = result.get(
                "f1"
            )

            error = result.get(
                "error"
            )

            if error:

                issue = shorten(
                    error,
                    100,
                )

            elif (
                answer_type
                == "insufficient_evidence"
                and result.get(
                    "is_answerable"
                ) is True
            ):

                issue = (
                    "Returned insufficient evidence "
                    "for an answerable question"
                )

            else:

                issue = (
                    "Incorrect or partially correct "
                    "answer"
                )

            report.append(
                f"| {question_id} | "
                f"{answer_type} | "
                f"{format_value(em)} | "
                f"{format_value(f1)} | "
                f"{issue} |\n"
            )

    # --------------------------------------------------------
    # Final conclusion
    # --------------------------------------------------------

    report.append(
        "\n## 7. Conclusion\n"
    )

    report.append(
        "The benchmark results should be interpreted "
        "by separating retrieval performance from "
        "end-to-end answer quality. Strong retrieval "
        "metrics indicate that the system is generally "
        "able to locate relevant evidence, while lower "
        "EM and F1 indicate that additional improvements "
        "are required in reasoning, calculation, answer "
        "extraction, and/or formatting.\n"
    )

    # --------------------------------------------------------
    # Write file
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "".join(report)
        )

    print(
        f"Report generated successfully:\n"
        f"{OUTPUT_FILE}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    generate_report()

    """
python3 services/eval-service/generate_report.py


sudo chown -R $USER:$USER services/eval-service/results

It will read:

services/eval-service/results/benchmark_results.json


and create:

services/eval-service/results/benchmark_report.md


"""
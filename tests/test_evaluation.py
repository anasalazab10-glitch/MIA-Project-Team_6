"""Unit tests for the evaluation metrics and benchmark functions."""

from src.evaluation import compute_retrieval_metrics


def test_compute_retrieval_metrics_exact_hit():
    retrieved = ["doc1_chunk1", "doc1_chunk2", "doc1_chunk3"]
    relevant = ["doc1_chunk1"]

    metrics = compute_retrieval_metrics(retrieved, relevant, k_list=[1, 3, 5])

    assert metrics["rr"] == 1.0  # Rank 1 hit
    assert metrics["hit@1"] == 1.0
    assert metrics["hit@3"] == 1.0
    assert metrics["recall@1"] == 1.0
    assert metrics["precision@1"] == 1.0


def test_compute_retrieval_metrics_delayed_hit():
    retrieved = ["irrelevant1", "irrelevant2", "target_chunk"]
    relevant = ["target_chunk"]

    metrics = compute_retrieval_metrics(retrieved, relevant, k_list=[1, 3, 5])

    assert metrics["rr"] == 1.0 / 3.0  # Rank 3 hit
    assert metrics["hit@1"] == 0.0
    assert metrics["hit@3"] == 1.0
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@3"] == 1.0


def test_compute_retrieval_metrics_zero_hit():
    retrieved = ["irrelevant1", "irrelevant2"]
    relevant = ["missing_chunk"]

    metrics = compute_retrieval_metrics(retrieved, relevant, k_list=[1, 3, 5])

    assert metrics["rr"] == 0.0
    assert metrics["hit@1"] == 0.0
    assert metrics["hit@3"] == 0.0
    assert metrics["recall@3"] == 0.0

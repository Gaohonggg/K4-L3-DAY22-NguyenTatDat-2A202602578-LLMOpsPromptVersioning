"""Dataset validation, strict score aggregation, and atomic JSON checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import config  # Set tracing configuration before importing RAGAS/LangChain.

from ragas import EvaluationDataset, SingleTurnSample


METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")


def write_json(path: Path, value: Any) -> None:
    """Replace a JSON file atomically; reject NaN/Infinity instead of emitting them."""
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def validate_contexts(contexts: Any) -> list[str]:
    """Require individual nonempty passages, never a concatenated string."""
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("contexts must be a nonempty list[str]")
    if any(not isinstance(item, str) or not item.strip() for item in contexts):
        raise ValueError("Each retrieved context must be a nonempty string")
    return contexts


def build_ragas_dataset(records: list[dict[str, Any]]) -> EvaluationDataset:
    """Map the four required fields to SingleTurnSample without using references in generation."""
    if not records:
        raise ValueError("Cannot evaluate an empty dataset")
    samples = []
    for record in records:
        for field in ("question", "answer", "reference"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"Missing/nonempty string required for {field}")
        samples.append(SingleTurnSample(
            user_input=record["question"],
            response=record["answer"],
            retrieved_contexts=validate_contexts(record.get("contexts")),
            reference=record["reference"],
        ))
    return EvaluationDataset(samples=samples)


def normalize_scores(rows: list[dict[str, Any]], expected_count: int) -> list[dict[str, float]]:
    """Reject missing samples or metrics; never hide evaluator failures in a mean."""
    if expected_count <= 0 or len(rows) != expected_count:
        raise ValueError("Score count does not match the expected sample count")
    normalized = []
    for index, row in enumerate(rows):
        sample = {}
        for metric in METRIC_NAMES:
            value = row.get(metric)
            if value is None or isinstance(value, bool):
                raise ValueError(f"Missing/invalid {metric} at sample {index + 1}")
            score = float(value)
            if not math.isfinite(score):
                raise ValueError(f"Nonfinite {metric} at sample {index + 1}")
            sample[metric] = score
        normalized.append(sample)
    return normalized


def summarize_scores(rows: list[dict[str, Any]], expected_count: int) -> dict[str, float]:
    """Compute each metric's mean over every expected sample."""
    normalized = normalize_scores(rows, expected_count)
    return {metric: mean(row[metric] for row in normalized) for metric in METRIC_NAMES}

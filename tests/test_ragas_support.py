"""Offline checks for RAGAS input integrity, score completeness, and report modes."""

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.ragas_support import (
    METRIC_NAMES,
    build_ragas_dataset,
    normalize_scores,
    summarize_scores,
    write_json,
)


class RagasSupportTests(unittest.TestCase):
    """Protect against silent sample loss and invalid evidence without API calls."""

    def setUp(self):
        self.record = {
            "question": "What is RAG?",
            "answer": "Retrieval-augmented generation.",
            "reference": "RAG combines retrieval and generation.",
            "contexts": ["RAG combines retrieval and generation.", "The answer uses retrieved context."],
        }
        self.scores = {metric: 0.9 for metric in METRIC_NAMES}

    def test_dataset_preserves_all_four_fields(self):
        sample = build_ragas_dataset([self.record]).samples[0]
        self.assertEqual(sample.user_input, self.record["question"])
        self.assertEqual(sample.response, self.record["answer"])
        self.assertEqual(sample.reference, self.record["reference"])
        self.assertEqual(sample.retrieved_contexts, self.record["contexts"])

    def test_concatenated_or_empty_contexts_are_rejected(self):
        for contexts in ("one combined string", [], [""], [None]):
            with self.subTest(contexts=contexts), self.assertRaises(ValueError):
                build_ragas_dataset([{**self.record, "contexts": contexts}])

    def test_empty_required_fields_are_rejected(self):
        for field in ("question", "answer", "reference"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_ragas_dataset([{**self.record, field: ""}])

    def test_incomplete_score_count_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_scores([self.scores], expected_count=50)

    def test_invalid_or_missing_scores_are_rejected(self):
        for value in (None, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_scores([{**self.scores, "faithfulness": value}], 1)
        with self.assertRaises(ValueError):
            normalize_scores([{"faithfulness": 0.9}], 1)

    def test_means_include_every_sample(self):
        rows = [{metric: value for metric in METRIC_NAMES} for value in (0.0, 0.5, 1.0)]
        self.assertEqual(summarize_scores(rows, 3), {metric: 0.5 for metric in METRIC_NAMES})

    def test_json_rejects_nan_without_overwriting_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            write_json(path, {"completed": 5})
            with self.assertRaises(ValueError):
                write_json(path, {"score": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"completed": 5})

    def test_smoke_report_is_never_submission_ready(self):
        module = importlib.import_module("03_ragas_evaluation")
        state = {
            "settings": {"sample_count_per_version": 2},
            "outputs": {tag: [self.record, self.record] for tag in ("v1", "v2")},
            "scores": {tag: [self.scores, self.scores] for tag in ("v1", "v2")},
        }
        report = module.make_report(state, smoke=True)
        self.assertTrue(report["target_met"])
        self.assertFalse(report["submission_ready"])
        self.assertEqual(report["evaluated_samples_total"], 4)


if __name__ == "__main__":
    unittest.main()

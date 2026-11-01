import unittest
from pathlib import Path

from backend.evaluation import evaluate_matcher, load_evaluation_dataset
from backend.knowledge_matcher import KnowledgeMatcher, load_dataset


ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def test_metrics_are_computed_from_labeled_evaluation_cases(self):
        matcher = KnowledgeMatcher(load_dataset(ROOT / "training" / "alert_rules.json"))
        cases = load_evaluation_dataset(ROOT / "training" / "evaluation_cases.json")
        result = evaluate_matcher(matcher, cases)

        self.assertEqual(result["case_count"], 12)
        self.assertEqual(result["true_positive"], 5)
        self.assertGreaterEqual(result["false_positive"], 0)
        self.assertGreaterEqual(result["false_negative"], 0)
        self.assertIsNotNone(result["precision"])
        self.assertIsNotNone(result["recall"])
        self.assertGreaterEqual(result["matcher_latency_ms_per_case"], 0)


if __name__ == "__main__":
    unittest.main()
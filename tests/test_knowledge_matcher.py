import json
import tempfile
import unittest
from pathlib import Path

from backend.knowledge_matcher import (
    DatasetValidationError,
    KnowledgeMatcher,
    load_dataset,
    validate_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "training" / "alert_rules.json"


class KnowledgeMatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset(DATASET_PATH)
        cls.matcher = KnowledgeMatcher(cls.dataset)

    def match(self, text, start=4.2):
        return self.matcher.match_segments([{"start": start, "end": start + 1, "text": text}])

    def test_exact_phrase_retains_original_evidence_and_timestamp(self):
        result = self.match("Please help!")[0]
        self.assertEqual(result["matched_phrase"], "please help")
        self.assertEqual(result["original_evidence"], "Please help!")
        self.assertEqual(result["start"], 4.2)
        self.assertEqual(result["priority"], "HIGH")
        self.assertEqual(result["match_type"], "exact")

    def test_configured_priority_is_preserved(self):
        result = self.match("Stop the machine")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["priority"], "MEDIUM")

    def test_phrase_boundaries_avoid_substring_match(self):
        self.assertEqual(self.match("The helper arrived."), [])

    def test_case_punctuation_and_whitespace_are_normalized(self):
        result = self.match("  THERE... IS   SMOKE! ")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["event_id"], "hazard_report")

    def test_negated_fire_report_does_not_match(self):
        self.assertEqual(self.match("There is no fire."), [])

    def test_negation_is_respected_when_split_across_transcript_segments(self):
        result = self.matcher.match_segments([
            {"start": 1, "end": 1.2, "text": "There"},
            {"start": 1.2, "end": 1.4, "text": "is no"},
            {"start": 1.4, "end": 1.8, "text": "fire"},
        ])
        self.assertEqual(result, [])

    def test_phrase_split_across_segments_preserves_joined_evidence(self):
        result = self.matcher.match_segments([
            {"start": 2, "end": 2.2, "text": "Please"},
            {"start": 2.2, "end": 2.6, "text": "help"},
        ])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["original_evidence"], "Please help")
        self.assertEqual(result[0]["start"], 2.2)

    def test_fictional_fire_context_needs_verification_without_alert(self):
        result = self.match("A fire scene in a movie.")[0]
        self.assertEqual(result["status"], "needs_verification")
        self.assertFalse(result["is_alert"])
        self.assertIn("Needs verification", result["explanation"])

    def test_hindi_phrase_variant_matches(self):
        result = self.match("कृपया मदद करें।")[0]
        self.assertEqual(result["event_id"], "assistance_request")
        self.assertEqual(result["matched_phrase"], "कृपया मदद करें")

    def test_duplicate_segments_at_same_timestamp_are_deduplicated(self):
        segment = {"start": 1, "end": 2, "text": "Help me"}
        result = self.matcher.match_segments([segment, dict(segment)])
        self.assertEqual(len(result), 1)

    def test_repeated_phrase_at_distinct_times_is_preserved(self):
        result = self.matcher.match_segments([
            {"start": 1, "end": 2, "text": "Help me"},
            {"start": 5, "end": 6, "text": "Help me"},
        ])
        self.assertEqual([item["start"] for item in result], [1.0, 5.0])

    def test_fuzzy_match_is_opt_in_and_score_is_not_confidence(self):
        segment = [{"start": 0, "end": 1, "text": "Please halp me"}]
        self.assertEqual(self.matcher.match_segments(segment), [])
        fuzzy = KnowledgeMatcher(self.dataset, fuzzy_enabled=True).match_segments(segment)
        self.assertEqual(fuzzy[0]["match_type"], "fuzzy")
        self.assertGreaterEqual(fuzzy[0]["match_score"], 0.88)

    def test_dataset_rejects_duplicate_ids_and_invalid_priority(self):
        duplicate = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        duplicate["rules"].append(dict(duplicate["rules"][0]))
        with self.assertRaisesRegex(DatasetValidationError, "Duplicate event_id"):
            validate_dataset(duplicate)

        invalid = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        invalid["rules"][0]["priority"] = "CRITICAL"
        with self.assertRaisesRegex(DatasetValidationError, "priority"):
            validate_dataset(invalid)

    def test_loader_reports_corrupted_json(self):
        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "broken.json"
            broken.write_text("{broken", encoding="utf-8")
            with self.assertRaises(DatasetValidationError):
                load_dataset(broken)


if __name__ == "__main__":
    unittest.main()
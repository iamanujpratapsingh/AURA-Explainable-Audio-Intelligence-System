import unittest
import time
from pathlib import Path

from backend.knowledge_matcher import KnowledgeMatcher, load_dataset
from backend.live_pipeline import LiveTranscriptProcessor, monitoring_state


ROOT = Path(__file__).resolve().parents[1]


class LiveTranscriptProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset = load_dataset(ROOT / "training" / "alert_rules.json")
        cls.matcher = KnowledgeMatcher(dataset)

    def make_event(self, text, event_id, capture_seconds=1.0, capture_monotonic=None):
        now = time.monotonic()
        return {
            "kind": "final",
            "event_id": event_id,
            "text": text,
            "capture_seconds": capture_seconds,
            "capture_monotonic": capture_monotonic if capture_monotonic is not None else now - 0.1,
            "recognized_monotonic": now,
        }

    def test_monitoring_states_require_live_samples_and_surface_errors(self):
        self.assertEqual(monitoring_state(False, False, False, "stopped"), "Idle")
        self.assertEqual(monitoring_state(True, False, False, "connecting"), "Processing")
        self.assertEqual(monitoring_state(True, True, True, "waiting_for_microphone"), "Processing")
        self.assertEqual(monitoring_state(True, True, False, "connected"), "Processing")
        self.assertEqual(monitoring_state(True, True, True, "connected"), "Listening")
        self.assertEqual(monitoring_state(True, False, True, "connected", paused=True), "Paused")
        self.assertEqual(monitoring_state(True, True, True, "error", error=True), "Error")

    def test_exact_live_phrase_matches_with_measured_latency(self):
        processor = LiveTranscriptProcessor(self.matcher)
        result = processor.process(self.make_event("Please help!", 1, capture_seconds=2.4))
        self.assertEqual(len(result["detections"]), 1)
        self.assertEqual(result["detections"][0]["original_evidence"], "Please help!")
        self.assertEqual(result["segment"]["start"], 2.4)
        self.assertGreaterEqual(result["capture_to_match_ms"], 0)
        self.assertGreaterEqual(result["recognition_to_match_ms"], 0)
        self.assertLessEqual(result["match_started_monotonic"], result["matched_monotonic"])

    def test_ui_latency_helper_handles_missing_timestamps(self):
        self.assertIsNone(LiveTranscriptProcessor.elapsed_ms(None, time.monotonic()))
        self.assertGreaterEqual(
            LiveTranscriptProcessor.elapsed_ms(time.monotonic() - 0.01, time.monotonic()),
            0,
        )

    def test_duplicate_event_id_is_suppressed_but_repeated_utterance_is_kept(self):
        processor = LiveTranscriptProcessor(self.matcher)
        first = self.make_event("Help me", 11, 1.0)
        repeated = self.make_event("Help me", 12, 4.0)
        self.assertIsNotNone(processor.process(first))
        self.assertIsNone(processor.process(first))
        self.assertIsNotNone(processor.process(repeated))
        self.assertEqual(len(processor.snapshot()), 2)

    def test_negated_statement_does_not_create_hazard_detection(self):
        processor = LiveTranscriptProcessor(self.matcher)
        processor.process(self.make_event("There is no", 20, 1.0))
        result = processor.process(self.make_event("fire", 21, 1.4))
        self.assertEqual(result["detections"], [])

    def test_unknown_speech_is_preserved_without_detection(self):
        processor = LiveTranscriptProcessor(self.matcher)
        result = processor.process(self.make_event("The weather is clear", 30))
        self.assertEqual(result["detections"], [])
        self.assertEqual(result["segment"]["text"], "The weather is clear")
        self.assertEqual(result["segment"]["analysis_status"], "no_configured_match")
        self.assertIn("no phrase match", result["segment"]["analysis_explanation"])

    def test_finalized_phrase_reports_analysis_result(self):
        processor = LiveTranscriptProcessor(self.matcher)
        result = processor.process(self.make_event("Please help!", 31))
        self.assertEqual(result["segment"]["analysis_status"], "matched_configured_phrase")
        self.assertEqual(result["segment"]["matched_detection_count"], 1)

    def test_finalized_segments_keep_chronological_timestamps(self):
        processor = LiveTranscriptProcessor(self.matcher)
        processor.process(self.make_event("First", 31, capture_seconds=1.0))
        processor.process(self.make_event("Second", 32, capture_seconds=2.5))
        self.assertEqual(
            [item["start"] for item in processor.snapshot()],
            [1.0, 2.5],
        )

    def test_interim_revision_is_not_committed_as_a_segment(self):
        processor = LiveTranscriptProcessor(self.matcher)
        self.assertIsNone(processor.process({"kind": "interim", "text": "help me"}))
        self.assertEqual(processor.snapshot(), [])


if __name__ == "__main__":
    unittest.main()

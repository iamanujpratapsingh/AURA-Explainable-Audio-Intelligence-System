import json
import time
from pathlib import Path

from backend.knowledge_matcher import DatasetValidationError, KnowledgeMatcher


def load_evaluation_dataset(path):
    try:
        with Path(path).open("r", encoding="utf-8") as data_file:
            data = json.load(data_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"Could not load evaluation dataset: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise DatasetValidationError("Evaluation dataset must contain a cases list.")
    case_ids = set()
    for index, case in enumerate(data["cases"]):
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise DatasetValidationError(f"Evaluation case at index {index} must have a case_id.")
        if case["case_id"] in case_ids:
            raise DatasetValidationError(f"Duplicate evaluation case_id: {case['case_id']}.")
        case_ids.add(case["case_id"])
        if not isinstance(case.get("transcript"), str) or not isinstance(case.get("expected_event_ids"), list):
            raise DatasetValidationError(f"Evaluation case {case['case_id']!r} has invalid transcript or labels.")
    return data


def evaluate_matcher(matcher, evaluation_dataset):
    event_ids = {rule["event_id"] for rule in matcher.dataset["rules"]}
    true_positive = false_positive = false_negative = true_negative = 0
    elapsed_ns = 0
    case_results = []

    for case in evaluation_dataset["cases"]:
        expected = set(case["expected_event_ids"])
        unknown = expected - event_ids
        if unknown:
            raise DatasetValidationError(
                f"Evaluation case {case['case_id']!r} references unknown rules: {', '.join(sorted(unknown))}."
            )
        start_time = time.perf_counter_ns()
        detections = matcher.match_segments([{"start": 0.0, "end": 0.0, "text": case["transcript"]}])
        elapsed_ns += time.perf_counter_ns() - start_time
        predicted = {item["event_id"] for item in detections if item["is_alert"]}
        true_positive += len(expected & predicted)
        false_positive += len(predicted - expected)
        false_negative += len(expected - predicted)
        true_negative += len(event_ids - (expected | predicted))
        case_results.append({
            "case_id": case["case_id"],
            "expected_event_ids": sorted(expected),
            "predicted_event_ids": sorted(predicted),
            "review_only_event_ids": sorted(
                item["event_id"] for item in detections if not item["is_alert"]
            ),
        })

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "case_count": len(evaluation_dataset["cases"]),
        "event_class_count": len(event_ids),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive / (false_positive + true_negative) if false_positive + true_negative else None,
        "missed_event_rate": false_negative / (true_positive + false_negative) if true_positive + false_negative else None,
        "matcher_latency_ms_per_case": elapsed_ns / 1_000_000 / len(evaluation_dataset["cases"]) if evaluation_dataset["cases"] else None,
        "case_results": case_results,
    }
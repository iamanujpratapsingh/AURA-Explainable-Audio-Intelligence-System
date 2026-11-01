import hashlib
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


PRIORITIES = {"LOW", "MEDIUM", "HIGH"}
REQUIRED_FIELDS = {
    "event_id",
    "event_name",
    "trigger_phrases",
    "phrase_variants",
    "language",
    "category",
    "event_type",
    "priority",
    "explanation_template",
    "recommended_action",
    "evidence_requirements",
    "source_reference",
    "requires_human_verification",
}


class DatasetValidationError(ValueError):
    pass


def normalize_text(text):
    text = str(text or "").casefold()
    text = "".join(
        character
        if character.isspace() or character.isalnum() or character == "_" or unicodedata.category(character).startswith("M")
        else " "
        for character in text
    )
    return " ".join(text.split())


def _phrase_matches(text, phrase):
    normalized_text = normalize_text(text)
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    return re.search(
        rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)",
        normalized_text,
        flags=re.UNICODE,
    ) is not None


def _string_list(value, field, event_id):
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise DatasetValidationError(f"Rule {event_id!r}: {field} must be a list of non-empty strings.")
    return [item.strip() for item in value]


def validate_dataset(data):
    if not isinstance(data, dict) or not isinstance(data.get("rules"), list):
        raise DatasetValidationError("Dataset must be an object containing a rules list.")

    matching = data.get("matching", {})
    if not isinstance(matching, dict):
        raise DatasetValidationError("Dataset matching settings must be an object.")
    threshold = matching.get("fuzzy_threshold", 0.88)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0 <= threshold <= 1:
        raise DatasetValidationError("matching.fuzzy_threshold must be a number from 0 to 1.")
    if not isinstance(matching.get("semantic_matching", False), bool):
        raise DatasetValidationError("matching.semantic_matching must be boolean.")

    ids = set()
    rules = []
    for index, raw_rule in enumerate(data["rules"]):
        if not isinstance(raw_rule, dict):
            raise DatasetValidationError(f"Rule at index {index} must be an object.")
        event_id = raw_rule.get("event_id")
        missing = REQUIRED_FIELDS - raw_rule.keys()
        if missing:
            raise DatasetValidationError(f"Rule {event_id or index!r} is missing: {', '.join(sorted(missing))}.")
        if not isinstance(event_id, str) or not event_id.strip():
            raise DatasetValidationError(f"Rule at index {index} must have a non-empty event_id.")
        if event_id in ids:
            raise DatasetValidationError(f"Duplicate event_id: {event_id}.")
        ids.add(event_id)
        if str(raw_rule["priority"]).upper() not in PRIORITIES:
            raise DatasetValidationError(f"Rule {event_id!r}: priority must be LOW, MEDIUM, or HIGH.")
        if not isinstance(raw_rule["requires_human_verification"], bool):
            raise DatasetValidationError(f"Rule {event_id!r}: requires_human_verification must be boolean.")
        for field in ("event_name", "language", "category", "event_type", "explanation_template", "recommended_action", "source_reference"):
            if not isinstance(raw_rule[field], str) or not raw_rule[field].strip():
                raise DatasetValidationError(f"Rule {event_id!r}: {field} must be a non-empty string.")
        rule = dict(raw_rule)
        for field in ("trigger_phrases", "phrase_variants", "evidence_requirements"):
            rule[field] = _string_list(raw_rule[field], field, event_id)
        for field in ("negative_phrases", "context_exclusions", "ambiguity_phrases"):
            if field in raw_rule:
                rule[field] = _string_list(raw_rule[field], field, event_id)
            else:
                rule[field] = []
        if not rule["trigger_phrases"] and not rule["phrase_variants"]:
            raise DatasetValidationError(f"Rule {event_id!r} must define at least one trigger phrase or variant.")
        rule["priority"] = str(rule["priority"]).upper()
        rules.append(rule)

    return {
        "matching": {
            "fuzzy_threshold": float(threshold),
            "semantic_matching": matching.get("semantic_matching", False),
        },
        "rules": rules,
    }


def load_dataset(path):
    try:
        with Path(path).open("r", encoding="utf-8") as dataset_file:
            data = json.load(dataset_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"Could not load knowledge dataset: {exc}") from exc
    return validate_dataset(data)


class KnowledgeMatcher:
    def __init__(self, dataset, fuzzy_enabled=False, fuzzy_threshold=None):
        self.dataset = validate_dataset(dataset)
        configured = self.dataset["matching"]["fuzzy_threshold"]
        self.fuzzy_enabled = fuzzy_enabled
        self.fuzzy_threshold = configured if fuzzy_threshold is None else fuzzy_threshold
        if not isinstance(self.fuzzy_threshold, (int, float)) or not 0 <= self.fuzzy_threshold <= 1:
            raise ValueError("fuzzy_threshold must be a number from 0 to 1.")

    @staticmethod
    def _fuzzy_candidate(text, phrases):
        tokens = normalize_text(text).split()
        best = (0.0, "")
        for phrase in phrases:
            phrase_tokens = normalize_text(phrase).split()
            if not phrase_tokens:
                continue
            minimum = max(1, len(phrase_tokens) - 1)
            maximum = min(len(tokens), len(phrase_tokens) + 1)
            for size in range(minimum, maximum + 1):
                for start in range(max(0, len(tokens) - size + 1)):
                    candidate = " ".join(tokens[start : start + size])
                    score = SequenceMatcher(None, normalize_text(phrase), candidate).ratio()
                    if score > best[0]:
                        best = (score, phrase)
        return best

    def match_segments(self, segments):
        detections = []
        seen = set()
        for segment_index, segment in enumerate(segments or []):
            if not isinstance(segment, dict):
                continue
            original_text = str(segment.get("text") or "")
            if not original_text.strip():
                continue
            prior_segments = segments[max(0, segment_index - 3) : segment_index]
            prior_text = " ".join(
                str(item.get("text") or "") for item in prior_segments if isinstance(item, dict)
            )
            context_text = " ".join(part for part in (prior_text, original_text) if part.strip())
            start = segment.get("start")
            end = segment.get("end", start)
            try:
                start = max(0.0, float(start or 0))
                end = max(start, float(end if end is not None else start))
            except (TypeError, ValueError):
                start, end = 0.0, 0.0

            for rule in self.dataset["rules"]:
                phrases = rule["trigger_phrases"] + rule["phrase_variants"]
                matched_phrase = next((p for p in phrases if _phrase_matches(original_text, p)), None)
                evidence_text = original_text
                if matched_phrase is None:
                    matched_in_context = next(
                        (p for p in phrases if _phrase_matches(context_text, p) and not _phrase_matches(prior_text, p)),
                        None,
                    )
                    if matched_in_context:
                        matched_phrase = matched_in_context
                        evidence_text = context_text
                match_type = "exact"
                score = 1.0
                if matched_phrase is None and self.fuzzy_enabled:
                    score, matched_phrase = self._fuzzy_candidate(original_text, phrases)
                    if score < self.fuzzy_threshold:
                        continue
                    match_type = "fuzzy"
                elif matched_phrase is None:
                    continue

                if any(_phrase_matches(context_text, phrase) for phrase in rule["negative_phrases"] + rule["context_exclusions"]):
                    continue
                ambiguous = any(_phrase_matches(context_text, phrase) for phrase in rule["ambiguity_phrases"])
                detection_id = hashlib.sha256(
                    f"{rule['event_id']}|{normalize_text(evidence_text)}|{start:.2f}|{matched_phrase}".encode("utf-8")
                ).hexdigest()[:16]
                if detection_id in seen:
                    continue
                seen.add(detection_id)

                explanation = rule["explanation_template"].format(
                    phrase=matched_phrase,
                    timestamp=f"{start:.2f}s",
                )
                if ambiguous:
                    explanation = f"Needs verification. {explanation} The surrounding context may be fictional, quoted, or otherwise unclear."
                detections.append(
                    {
                        "detection_id": detection_id,
                        "event_id": rule["event_id"],
                        "event_name": rule["event_name"],
                        "matched_phrase": matched_phrase,
                        "original_evidence": evidence_text,
                        "start": start,
                        "end": end,
                        "language": rule["language"],
                        "category": rule["category"],
                        "event_type": rule["event_type"],
                        "priority": rule["priority"],
                        "match_type": match_type,
                        "match_score": round(score, 3),
                        "status": "needs_verification" if ambiguous or rule["requires_human_verification"] else "detected",
                        "is_alert": not ambiguous,
                        "explanation": explanation,
                        "recommended_action": rule["recommended_action"],
                        "evidence_requirements": rule["evidence_requirements"],
                        "source_reference": rule["source_reference"],
                        "requires_human_verification": rule["requires_human_verification"],
                        "segment_index": segment.get("segment_index", segment_index),
                    }
                )
        return detections
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exporters import write_canonical_transcript
from speaker_registry import recognize_speakers


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _transcript_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}_transcript.json")


def _load_transcript(source: Path) -> dict[str, Any]:
    path = _transcript_path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Canonical transcript not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _passes_core_gates(item: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    """Apply the conservative gates that are independent of cross-candidate uniqueness.

    pyannote can split one real person into multiple diarization IDs in the same
    recording. Therefore 'unique_best' is useful diagnostic metadata, but it must
    not be a hard identity gate. Each diarization ID is evaluated independently
    against the registry using absolute similarity, runner-up margin and speech
    quantity.
    """
    min_chunks = int(thresholds.get("min_chunks", 2))
    similarity_threshold = float(thresholds.get("similarity", 0.60))
    margin_threshold = float(thresholds.get("margin", 0.20))

    chunks = int(item.get("chunks") or 0)
    score = item.get("score")
    runner_up = item.get("runner_up_score")
    margin = item.get("margin")

    if chunks < min_chunks or score is None:
        return False
    if float(score) < similarity_threshold:
        return False
    if runner_up is not None:
        if margin is None or float(margin) < margin_threshold:
            return False
    return True


def _preserved_manual_results(transcript: dict[str, Any]) -> dict[str, dict[str, Any]]:
    preserved: dict[str, dict[str, Any]] = {}
    for item in (transcript.get("speaker_resolution") or {}).get("results") or []:
        if item.get("status") != "CONFIRMED" or not item.get("resolved_name"):
            continue
        speaker = str(item.get("diarization_speaker") or "")
        if speaker:
            preserved[speaker] = dict(item)
    return preserved


def recognize_with_split_support(
    source: Path,
    registry_path: Path,
) -> dict[str, Any]:
    """Recognize registry identities without forcing one diarization ID per person."""
    report = recognize_speakers(source, registry_path)
    thresholds = report.get("thresholds") or {}

    for item in report.get("results") or []:
        if item.get("status") == "KNOWN":
            continue
        if not item.get("matched_display_name"):
            continue
        if not _passes_core_gates(item, thresholds):
            continue

        item["status"] = "KNOWN"
        item["resolved_name"] = item.get("matched_display_name")
        item["reason"] = (
            "confident_match"
            if item.get("unique_best")
            else "confident_match_split_diarization"
        )

    return report


def apply_recognition(source: Path, registry_path: Path) -> dict[str, Any]:
    """Apply current registry recognition and regenerate outputs without retranscription.

    User-confirmed assignments already present in a transcript are preserved.
    Automatic recognition is recalculated for every other diarization speaker.
    """
    source = source.expanduser().resolve()
    transcript = _load_transcript(source)
    manual = _preserved_manual_results(transcript)
    report = recognize_with_split_support(source, registry_path)

    final_results: list[dict[str, Any]] = []
    for item in report.get("results") or []:
        speaker = str(item.get("diarization_speaker") or "")
        if speaker in manual:
            preserved = dict(manual[speaker])
            preserved["reason"] = preserved.get("reason") or "user_confirmed"
            final_results.append(preserved)
        else:
            final_results.append(item)

    # Keep a confirmed speaker even if it no longer appears in the fresh embedding
    # report (for example because it has too little usable speech in this recording).
    reported = {
        str(item.get("diarization_speaker") or "") for item in final_results
    }
    for speaker, item in manual.items():
        if speaker not in reported:
            final_results.append(item)

    final_results.sort(key=lambda item: str(item.get("diarization_speaker") or ""))
    report["results"] = final_results

    resolved_by_speaker = {
        str(item.get("diarization_speaker")): str(item.get("resolved_name"))
        for item in final_results
        if item.get("status") in {"KNOWN", "CONFIRMED"}
        and item.get("resolved_name")
    }

    updated_segments = 0
    for segment in transcript.get("segments") or []:
        speaker = str(segment.get("speaker") or "UNKNOWN")
        resolved_name = resolved_by_speaker.get(speaker)
        if segment.get("resolved_name") != resolved_name:
            updated_segments += 1
        segment["resolved_name"] = resolved_name

    compact_results = [
        {
            "diarization_speaker": item.get("diarization_speaker"),
            "status": item.get("status"),
            "resolved_name": item.get("resolved_name"),
            "matched_speaker_id": item.get("matched_speaker_id"),
            "matched_display_name": item.get("matched_display_name"),
            "score": item.get("score"),
            "runner_up_score": item.get("runner_up_score"),
            "margin": item.get("margin"),
            "unique_best": item.get("unique_best"),
            "chunks": item.get("chunks"),
            "reason": item.get("reason"),
            "voice_reference_saved": item.get("voice_reference_saved"),
        }
        for item in final_results
    ]

    transcript["speaker_resolution"] = {
        "resolved_at": _now(),
        "registry": report.get("registry"),
        "embedding_model": report.get("embedding_model"),
        "thresholds": report.get("thresholds") or {},
        "known_identity_count": report.get("known_identity_count", 0),
        "results": compact_results,
    }

    outputs = write_canonical_transcript(transcript, source)
    report["applied"] = True
    report["updated_segments"] = updated_segments
    report["resolved_speakers"] = resolved_by_speaker
    report["outputs"] = {key: str(path) for key, path in outputs.items()}
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run known-speaker recognition on an existing canonical transcript."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = apply_recognition(args.source, args.registry)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()

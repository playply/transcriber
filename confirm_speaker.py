from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exporters import write_canonical_transcript
from speaker_registry import enroll_speaker


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _transcript_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}_transcript.json")


def confirm_speaker(
    source: Path,
    diarization_speaker: str,
    display_name: str,
    registry_path: Path,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    diarization_speaker = diarization_speaker.strip()
    display_name = display_name.strip()
    if not diarization_speaker or diarization_speaker == "UNKNOWN":
        raise ValueError("Choose a concrete diarization speaker ID.")
    if not display_name:
        raise ValueError("Display name must not be empty.")

    enrollment = enroll_speaker(
        source,
        diarization_speaker,
        display_name,
        registry_path,
    )

    transcript_path = _transcript_path(source)
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))

    updated_segments = 0
    for segment in transcript.get("segments") or []:
        if str(segment.get("speaker") or "UNKNOWN") != diarization_speaker:
            continue
        if segment.get("resolved_name") != display_name:
            updated_segments += 1
        segment["resolved_name"] = display_name

    resolution = dict(transcript.get("speaker_resolution") or {})
    results = [
        item
        for item in resolution.get("results") or []
        if str(item.get("diarization_speaker")) != diarization_speaker
    ]
    results.append(
        {
            "diarization_speaker": diarization_speaker,
            "status": "CONFIRMED",
            "resolved_name": display_name,
            "matched_speaker_id": enrollment.get("speaker_id"),
            "matched_display_name": display_name,
            "score": None,
            "runner_up_score": None,
            "margin": None,
            "unique_best": True,
            "chunks": enrollment.get("chunks_added"),
            "reason": "user_confirmed",
        }
    )
    resolution["resolved_at"] = _now()
    resolution["registry"] = str(registry_path.expanduser().resolve())
    resolution["results"] = sorted(
        results,
        key=lambda item: str(item.get("diarization_speaker") or ""),
    )
    transcript["speaker_resolution"] = resolution

    outputs = write_canonical_transcript(transcript, source)
    return {
        **enrollment,
        "confirmed": True,
        "updated_segments": updated_segments,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confirm an unknown diarized speaker, save it to the registry, and regenerate transcript files."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("diarization_speaker")
    parser.add_argument("display_name")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = confirm_speaker(
        args.source,
        args.diarization_speaker,
        args.display_name,
        args.registry,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()

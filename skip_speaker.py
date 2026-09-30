from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exporters import write_canonical_transcript


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _transcript_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}_transcript.json")


def keep_unknown(source: Path, diarization_speaker: str) -> dict[str, Any]:
    source = source.expanduser().resolve()
    diarization_speaker = diarization_speaker.strip()

    if not diarization_speaker or diarization_speaker == "UNKNOWN":
        raise ValueError("Choose a concrete diarization speaker ID.")

    transcript_path = _transcript_path(source)
    if not transcript_path.is_file():
        raise FileNotFoundError(f"Canonical transcript not found: {transcript_path}")

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    matching_segments = [
        segment
        for segment in transcript.get("segments") or []
        if str(segment.get("speaker") or "UNKNOWN") == diarization_speaker
    ]
    if not matching_segments:
        raise ValueError(
            f"Diarization speaker not found in transcript: {diarization_speaker}"
        )
    if any(str(segment.get("resolved_name") or "").strip() for segment in matching_segments):
        raise ValueError(
            f"{diarization_speaker} is already resolved. Reload the transcript."
        )

    resolution = dict(transcript.get("speaker_resolution") or {})
    previous = None
    remaining: list[dict[str, Any]] = []
    for item in resolution.get("results") or []:
        if str(item.get("diarization_speaker") or "") == diarization_speaker:
            previous = dict(item)
        else:
            remaining.append(item)

    if previous and previous.get("status") in {"KNOWN", "CONFIRMED"}:
        raise ValueError(
            f"{diarization_speaker} is already resolved. Reload the transcript."
        )

    decision_at = _now()
    skipped = dict(previous or {})
    skipped.update(
        {
            "diarization_speaker": diarization_speaker,
            "status": "SKIPPED",
            "resolved_name": None,
            "reason": "user_kept_unknown",
            "voice_reference_saved": False,
            "user_decision": "keep_unknown",
            "user_decision_at": decision_at,
        }
    )
    remaining.append(skipped)
    resolution["resolved_at"] = decision_at
    resolution["results"] = sorted(
        remaining,
        key=lambda item: str(item.get("diarization_speaker") or ""),
    )
    transcript["speaker_resolution"] = resolution

    outputs = write_canonical_transcript(transcript, source)
    return {
        "source": str(source),
        "diarization_speaker": diarization_speaker,
        "status": "SKIPPED",
        "registry_changed": False,
        "voice_reference_saved": False,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep one diarized speaker unresolved for this recording without "
            "changing the persistent speaker registry."
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("diarization_speaker")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = keep_unknown(args.source, args.diarization_speaker)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()

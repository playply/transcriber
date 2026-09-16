from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exporters import write_canonical_transcript
from speaker_registry import enroll_speaker, load_registry, save_registry


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _transcript_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}_transcript.json")


def _find_identity_by_name(registry: dict[str, Any], display_name: str) -> dict[str, Any] | None:
    wanted = display_name.strip().casefold()
    for identity in registry.get("speakers") or []:
        if str(identity.get("display_name") or "").strip().casefold() == wanted:
            return identity
    return None


def _save_metadata_only_identity(
    source: Path,
    diarization_speaker: str,
    display_name: str,
    registry_path: Path,
    enrollment_error: str,
) -> dict[str, Any]:
    """Persist a user-confirmed identity without voice embeddings.

    This is used only when there is not enough usable speech to build a safe
    embedding reference. Such identities stay out of automatic voice matching
    until a later confirmation supplies sufficient speech.
    """
    registry = load_registry(registry_path)
    identity = _find_identity_by_name(registry, display_name)
    created = identity is None
    if identity is None:
        identity = {
            "speaker_id": "spk_" + uuid.uuid4().hex[:12],
            "display_name": display_name,
            "embeddings": [],
            "provenance": [],
            "created_at": _now(),
            "updated_at": _now(),
        }
        registry.setdefault("speakers", []).append(identity)

    identity["display_name"] = display_name
    identity["updated_at"] = _now()
    identity.setdefault("provenance", []).append(
        {
            "source": str(source),
            "transcript": str(_transcript_path(source)),
            "diarization_speaker": diarization_speaker,
            "chunks": 0,
            "added_at": _now(),
            "quality": "insufficient_usable_speech",
            "note": enrollment_error,
        }
    )
    save_registry(registry_path, registry)

    return {
        "registry": str(registry_path.expanduser().resolve()),
        "created": created,
        "speaker_id": identity["speaker_id"],
        "display_name": identity["display_name"],
        "source": str(source),
        "diarization_speaker": diarization_speaker,
        "chunks_added": 0,
        "reference_embeddings": len(identity.get("embeddings") or []),
        "voice_reference_saved": False,
        "voice_reference_note": enrollment_error,
    }


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

    try:
        enrollment = enroll_speaker(
            source,
            diarization_speaker,
            display_name,
            registry_path,
        )
        enrollment["voice_reference_saved"] = True
        enrollment["voice_reference_note"] = None
    except RuntimeError as exc:
        message = str(exc)
        if "does not have enough usable speech" not in message:
            raise
        enrollment = _save_metadata_only_identity(
            source,
            diarization_speaker,
            display_name,
            registry_path,
            message,
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
            "voice_reference_saved": enrollment.get("voice_reference_saved", True),
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

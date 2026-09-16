from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exporters import write_canonical_transcript
from speaker_registry import (
    MAX_EMBEDDINGS_PER_IDENTITY,
    MIN_CHUNKS,
    _source_embeddings,
    load_registry,
    save_registry,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _transcript_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}_transcript.json")


def _find_identity_by_id(
    registry: dict[str, Any], speaker_id: str
) -> dict[str, Any] | None:
    for identity in registry.get("speakers") or []:
        if str(identity.get("speaker_id") or "") == speaker_id:
            return identity
    return None


def _create_identity(display_name: str) -> dict[str, Any]:
    now = _now()
    return {
        "speaker_id": "spk_" + uuid.uuid4().hex[:12],
        "display_name": display_name,
        "embeddings": [],
        "provenance": [],
        "created_at": now,
        "updated_at": now,
    }


def _attach_reference(
    source: Path,
    diarization_speaker: str,
    identity: dict[str, Any],
    registry: dict[str, Any],
    registry_path: Path,
) -> dict[str, Any]:
    embeddings, meta = _source_embeddings(source)
    vectors = embeddings.get(diarization_speaker) or []
    enough_speech = len(vectors) >= MIN_CHUNKS

    identity["updated_at"] = _now()
    if enough_speech:
        refs = list(identity.get("embeddings") or [])
        refs.extend(vector.astype(float).tolist() for vector in vectors)
        identity["embeddings"] = refs[-MAX_EMBEDDINGS_PER_IDENTITY:]

    identity.setdefault("provenance", []).append(
        {
            "source": str(source),
            "transcript": meta["transcript"],
            "diarization_speaker": diarization_speaker,
            "chunks": len(vectors),
            "added_at": _now(),
            "quality": "voice_reference" if enough_speech else "insufficient_usable_speech",
        }
    )
    save_registry(registry_path, registry)

    return {
        "chunks_added": len(vectors) if enough_speech else 0,
        "reference_embeddings": len(identity.get("embeddings") or []),
        "voice_reference_saved": enough_speech,
        "voice_reference_note": (
            None
            if enough_speech
            else (
                f"{diarization_speaker} has {len(vectors)} usable chunks; "
                f"need at least {MIN_CHUNKS} for a voice reference."
            )
        ),
        "representative_segment": (
            meta.get("representative_segments") or {}
        ).get(diarization_speaker),
    }


def confirm_speaker(
    source: Path,
    diarization_speaker: str,
    registry_path: Path,
    *,
    display_name: str | None = None,
    existing_speaker_id: str | None = None,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    diarization_speaker = diarization_speaker.strip()
    display_name = (display_name or "").strip()
    existing_speaker_id = (existing_speaker_id or "").strip() or None

    if not diarization_speaker or diarization_speaker == "UNKNOWN":
        raise ValueError("Choose a concrete diarization speaker ID.")

    registry = load_registry(registry_path)
    if existing_speaker_id:
        identity = _find_identity_by_id(registry, existing_speaker_id)
        if identity is None:
            raise ValueError(
                f"Registry identity not found: {existing_speaker_id}. Reload known people and try again."
            )
        created = False
        linked_existing = True
        resolved_name = str(identity.get("display_name") or "").strip()
        if not resolved_name:
            raise RuntimeError(
                f"Registry identity {existing_speaker_id} has no display name."
            )
    else:
        if not display_name:
            raise ValueError(
                "Enter a name for a new person, or choose an existing person."
            )
        identity = _create_identity(display_name)
        registry.setdefault("speakers", []).append(identity)
        created = True
        linked_existing = False
        resolved_name = display_name

    reference = _attach_reference(
        source,
        diarization_speaker,
        identity,
        registry,
        registry_path,
    )

    transcript_path = _transcript_path(source)
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))

    updated_segments = 0
    for segment in transcript.get("segments") or []:
        if str(segment.get("speaker") or "UNKNOWN") != diarization_speaker:
            continue
        if segment.get("resolved_name") != resolved_name:
            updated_segments += 1
        segment["resolved_name"] = resolved_name

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
            "resolved_name": resolved_name,
            "matched_speaker_id": identity["speaker_id"],
            "matched_display_name": resolved_name,
            "score": None,
            "runner_up_score": None,
            "margin": None,
            "unique_best": True,
            "chunks": reference["chunks_added"],
            "reason": (
                "user_linked_existing_identity"
                if linked_existing
                else "user_created_identity"
            ),
            "voice_reference_saved": reference["voice_reference_saved"],
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
        "registry": str(registry_path.expanduser().resolve()),
        "created": created,
        "linked_existing": linked_existing,
        "speaker_id": identity["speaker_id"],
        "display_name": resolved_name,
        "source": str(source),
        "diarization_speaker": diarization_speaker,
        **reference,
        "confirmed": True,
        "updated_segments": updated_segments,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Confirm an unknown diarized speaker, link it to an existing registry identity "
            "or create a new identity, and regenerate transcript files."
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("diarization_speaker")
    parser.add_argument("--display-name")
    parser.add_argument("--existing-speaker-id")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = confirm_speaker(
        args.source,
        args.diarization_speaker,
        args.registry,
        display_name=args.display_name,
        existing_speaker_id=args.existing_speaker_id,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from exporters import write_canonical_transcript
from speaker_benchmark import (
    EMBEDDING_MODEL,
    _build_chunks,
    _cosine,
    _extract_embeddings,
    _normalize,
)

REGISTRY_VERSION = 1
SIMILARITY_THRESHOLD = 0.60
MARGIN_THRESHOLD = 0.20
MIN_CHUNKS = 2
MAX_EMBEDDINGS_PER_IDENTITY = 12


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_registry() -> dict[str, Any]:
    return {
        "version": REGISTRY_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "thresholds": {
            "similarity": SIMILARITY_THRESHOLD,
            "margin": MARGIN_THRESHOLD,
            "min_chunks": MIN_CHUNKS,
        },
        "speakers": [],
    }


def load_registry(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.exists():
        return _default_registry()

    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != REGISTRY_VERSION:
        raise RuntimeError(
            f"Unsupported speaker registry version: {data.get('version')}. Expected {REGISTRY_VERSION}."
        )
    if data.get("embedding_model") != EMBEDDING_MODEL:
        raise RuntimeError(
            "Speaker registry embedding model does not match the current model: "
            f"{data.get('embedding_model')} != {EMBEDDING_MODEL}"
        )
    if not isinstance(data.get("speakers"), list):
        raise RuntimeError("Speaker registry is malformed: speakers must be a list.")
    return data


def save_registry(path: Path, registry: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _transcript_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}_transcript.json")


def _load_transcript(source: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source recording not found: {source}")
    transcript_path = _transcript_path(source)
    if not transcript_path.is_file():
        raise FileNotFoundError(
            f"Canonical transcript not found: {transcript_path}. Transcribe the recording first."
        )
    return json.loads(transcript_path.read_text(encoding="utf-8"))


def _representative_segments(segments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for segment in segments:
        speaker = segment.get("speaker") or "UNKNOWN"
        if speaker == "UNKNOWN":
            continue
        start = segment.get("start")
        end = segment.get("end")
        if start is None or end is None:
            continue
        duration = float(end) - float(start)
        current = best.get(speaker)
        if current is None or duration > float(current["duration"]):
            best[speaker] = {
                "start": float(start),
                "end": float(end),
                "duration": duration,
                "text": str(segment.get("text") or "").strip()[:220],
            }
    return best


def _source_embeddings(source: Path) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    source = source.expanduser().resolve()
    transcript = _load_transcript(source)
    segments = transcript.get("segments") or []
    chunks = _build_chunks(segments)
    embeddings = _extract_embeddings(source, chunks) if chunks else {}
    detected = sorted(
        {
            str(segment.get("speaker"))
            for segment in segments
            if segment.get("speaker") and segment.get("speaker") != "UNKNOWN"
        }
    )
    return embeddings, {
        "source": str(source),
        "transcript": str(_transcript_path(source)),
        "detected_speakers": detected,
        "chunks_per_speaker": {
            speaker: len(vectors) for speaker, vectors in embeddings.items()
        },
        "representative_segments": _representative_segments(segments),
    }


def _identity_prototype(identity: dict[str, Any]) -> np.ndarray:
    refs = identity.get("embeddings") or []
    if not refs:
        raise RuntimeError(f"Registry identity {identity.get('speaker_id')} has no embeddings.")
    vectors = np.stack(
        [_normalize(np.asarray(vector, dtype=np.float32)) for vector in refs]
    )
    return _normalize(np.mean(vectors, axis=0))


def _candidate_prototypes(
    embeddings: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        speaker: _normalize(np.mean(np.stack(vectors), axis=0))
        for speaker, vectors in embeddings.items()
        if len(vectors) >= MIN_CHUNKS
    }


def _find_identity_by_name(
    registry: dict[str, Any], display_name: str
) -> dict[str, Any] | None:
    wanted = display_name.strip().casefold()
    for identity in registry["speakers"]:
        if str(identity.get("display_name") or "").strip().casefold() == wanted:
            return identity
    return None


def enroll_speaker(
    source: Path,
    diarization_speaker: str,
    display_name: str,
    registry_path: Path,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    display_name = display_name.strip()
    diarization_speaker = diarization_speaker.strip()
    if not display_name:
        raise ValueError("Display name must not be empty.")
    if not diarization_speaker or diarization_speaker == "UNKNOWN":
        raise ValueError("Choose a concrete diarization speaker ID.")

    embeddings, meta = _source_embeddings(source)
    vectors = embeddings.get(diarization_speaker)
    if not vectors or len(vectors) < MIN_CHUNKS:
        raise RuntimeError(
            f"{diarization_speaker} does not have enough usable speech. "
            f"Need at least {MIN_CHUNKS} chunks."
        )

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
        registry["speakers"].append(identity)

    identity["display_name"] = display_name
    identity["updated_at"] = _now()
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
        "chunks_added": len(vectors),
        "reference_embeddings": len(identity["embeddings"]),
        "representative_segment": (
            meta.get("representative_segments") or {}
        ).get(diarization_speaker),
    }


def recognize_speakers(source: Path, registry_path: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    registry = load_registry(registry_path)
    embeddings, meta = _source_embeddings(source)
    candidates = _candidate_prototypes(embeddings)

    identities = [
        identity for identity in registry["speakers"] if identity.get("embeddings")
    ]
    identity_prototypes = {
        str(identity["speaker_id"]): _identity_prototype(identity)
        for identity in identities
    }
    identity_by_id = {
        str(identity["speaker_id"]): identity for identity in identities
    }

    matrix: dict[str, dict[str, float]] = {}
    for diarization_speaker, candidate in sorted(candidates.items()):
        matrix[diarization_speaker] = {
            identity_id: _cosine(candidate, prototype)
            for identity_id, prototype in identity_prototypes.items()
        }

    best_candidate_for_identity: dict[str, str] = {}
    for identity_id in identity_prototypes:
        ranked = sorted(
            (
                (diarization_speaker, scores[identity_id])
                for diarization_speaker, scores in matrix.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if ranked:
            best_candidate_for_identity[identity_id] = ranked[0][0]

    results: list[dict[str, Any]] = []
    examples = meta.get("representative_segments") or {}
    chunks_per_speaker = meta.get("chunks_per_speaker") or {}

    for diarization_speaker in meta["detected_speakers"]:
        chunk_count = int(chunks_per_speaker.get(diarization_speaker, 0))
        scores = matrix.get(diarization_speaker) or {}
        if not scores:
            results.append(
                {
                    "diarization_speaker": diarization_speaker,
                    "resolved_name": None,
                    "status": "UNKNOWN",
                    "reason": (
                        "registry_empty"
                        if not identities
                        else "insufficient_usable_speech"
                    ),
                    "chunks": chunk_count,
                    "score": None,
                    "runner_up_score": None,
                    "margin": None,
                    "unique_best": False,
                    "representative_segment": examples.get(diarization_speaker),
                }
            )
            continue

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_identity_id, best_score = ranked[0]
        runner_up_score = ranked[1][1] if len(ranked) > 1 else None
        margin = (
            best_score - runner_up_score if runner_up_score is not None else None
        )
        unique_best = (
            best_candidate_for_identity.get(best_identity_id)
            == diarization_speaker
        )
        similarity_ok = best_score >= SIMILARITY_THRESHOLD
        margin_ok = (
            runner_up_score is None
            or (margin is not None and margin >= MARGIN_THRESHOLD)
        )
        known = (
            chunk_count >= MIN_CHUNKS
            and similarity_ok
            and margin_ok
            and unique_best
        )

        reasons: list[str] = []
        if chunk_count < MIN_CHUNKS:
            reasons.append("insufficient_usable_speech")
        if not similarity_ok:
            reasons.append("score_below_threshold")
        if not margin_ok:
            reasons.append("margin_below_threshold")
        if not unique_best:
            reasons.append("not_unique_best")

        identity = identity_by_id[best_identity_id]
        results.append(
            {
                "diarization_speaker": diarization_speaker,
                "resolved_name": identity["display_name"] if known else None,
                "matched_speaker_id": best_identity_id,
                "matched_display_name": identity["display_name"],
                "status": "KNOWN" if known else "UNKNOWN",
                "reason": "confident_match" if known else ",".join(reasons),
                "chunks": chunk_count,
                "score": best_score,
                "runner_up_score": runner_up_score,
                "margin": margin,
                "unique_best": unique_best,
                "representative_segment": examples.get(diarization_speaker),
            }
        )

    return {
        "registry": str(registry_path.expanduser().resolve()),
        "embedding_model": EMBEDDING_MODEL,
        "thresholds": registry.get("thresholds") or {},
        "known_identity_count": len(identities),
        "source": str(source),
        "results": results,
    }


def apply_recognition(source: Path, registry_path: Path) -> dict[str, Any]:
    """Resolve confident names in canonical JSON and regenerate TXT/DOCX."""
    source = source.expanduser().resolve()
    report = recognize_speakers(source, registry_path)
    transcript = _load_transcript(source)

    known_by_speaker = {
        str(item["diarization_speaker"]): str(item["resolved_name"])
        for item in report["results"]
        if item.get("status") == "KNOWN" and item.get("resolved_name")
    }

    updated_segments = 0
    for segment in transcript.get("segments") or []:
        speaker = str(segment.get("speaker") or "UNKNOWN")
        resolved_name = known_by_speaker.get(speaker)
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
        }
        for item in report["results"]
    ]

    transcript["speaker_resolution"] = {
        "resolved_at": _now(),
        "registry": report["registry"],
        "embedding_model": report["embedding_model"],
        "thresholds": report["thresholds"],
        "known_identity_count": report["known_identity_count"],
        "results": compact_results,
    }

    outputs = write_canonical_transcript(transcript, source)
    report["applied"] = True
    report["updated_segments"] = updated_segments
    report["resolved_speakers"] = known_by_speaker
    report["outputs"] = {key: str(path) for key, path in outputs.items()}
    return report


def registry_overview(registry_path: Path) -> dict[str, Any]:
    registry = load_registry(registry_path)
    return {
        "registry": str(registry_path.expanduser().resolve()),
        "embedding_model": registry.get("embedding_model"),
        "thresholds": registry.get("thresholds") or {},
        "known_identity_count": len(registry.get("speakers") or []),
        "speakers": [
            {
                "speaker_id": identity.get("speaker_id"),
                "display_name": identity.get("display_name"),
                "reference_embeddings": len(identity.get("embeddings") or []),
                "sources": len(identity.get("provenance") or []),
            }
            for identity in registry.get("speakers") or []
        ],
    }


def _write_report(report: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent speaker registry utility.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    enroll = subparsers.add_parser("enroll", help="Save a confirmed speaker identity.")
    enroll.add_argument("source", type=Path)
    enroll.add_argument("diarization_speaker")
    enroll.add_argument("display_name")
    enroll.add_argument("--registry", type=Path, required=True)
    enroll.add_argument("--output", type=Path)

    recognize = subparsers.add_parser(
        "recognize", help="Preview matches against the registry."
    )
    recognize.add_argument("source", type=Path)
    recognize.add_argument("--registry", type=Path, required=True)
    recognize.add_argument("--output", type=Path)

    apply_parser = subparsers.add_parser(
        "apply",
        help="Apply confident registry matches to canonical JSON and regenerate TXT/DOCX.",
    )
    apply_parser.add_argument("source", type=Path)
    apply_parser.add_argument("--registry", type=Path, required=True)
    apply_parser.add_argument("--output", type=Path)

    show = subparsers.add_parser("list", help="Show registry metadata without embeddings.")
    show.add_argument("--registry", type=Path, required=True)
    show.add_argument("--output", type=Path)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "enroll":
        report = enroll_speaker(
            args.source,
            args.diarization_speaker,
            args.display_name,
            args.registry,
        )
    elif args.command == "recognize":
        report = recognize_speakers(args.source, args.registry)
    elif args.command == "apply":
        report = apply_recognition(args.source, args.registry)
    elif args.command == "list":
        report = registry_overview(args.registry)
    else:
        raise RuntimeError(f"Unsupported command: {args.command}")

    _write_report(report, args.output)


if __name__ == "__main__":
    main()

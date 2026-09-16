from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from speaker_benchmark import (
    EMBEDDING_MODEL,
    _build_chunks,
    _cosine,
    _extract_embeddings,
    _normalize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare speaker embeddings across two already-transcribed recordings."
    )
    parser.add_argument("reference", type=Path, help="Reference recording path")
    parser.add_argument("candidate", type=Path, help="Second recording path")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser.parse_args()


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
        text = str(segment.get("text") or "").strip()
        current = best.get(speaker)
        if current is None or duration > float(current["duration"]):
            best[speaker] = {
                "start": float(start),
                "end": float(end),
                "duration": duration,
                "text": text[:220],
            }
    return best


def _load_recording(source: Path) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source recording not found: {source}")

    transcript_path = source.with_name(f"{source.stem}_transcript.json")
    if not transcript_path.is_file():
        raise FileNotFoundError(
            f"Canonical transcript not found: {transcript_path}. Transcribe the recording first."
        )

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = transcript.get("segments") or []
    chunks = _build_chunks(segments)
    if not chunks:
        raise RuntimeError(
            f"No speaker has enough clean speech for cross-recording comparison: {source.name}"
        )

    embeddings = _extract_embeddings(source, chunks)
    metadata = {
        "source": str(source),
        "transcript": str(transcript_path),
        "chunks_per_speaker": {
            speaker: len(vectors) for speaker, vectors in embeddings.items()
        },
        "representative_segments": _representative_segments(segments),
    }
    return embeddings, metadata


def _prototypes(embeddings: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        speaker: _normalize(np.mean(np.stack(vectors), axis=0))
        for speaker, vectors in embeddings.items()
    }


def run_cross_recording(reference: Path, candidate: Path) -> dict[str, Any]:
    reference = reference.expanduser().resolve()
    candidate = candidate.expanduser().resolve()
    if reference == candidate:
        raise ValueError("Reference and candidate recordings must be different files.")

    reference_embeddings, reference_meta = _load_recording(reference)
    candidate_embeddings, candidate_meta = _load_recording(candidate)

    reference_prototypes = _prototypes(reference_embeddings)
    candidate_prototypes = _prototypes(candidate_embeddings)

    matrix: dict[str, dict[str, float]] = {}
    for candidate_speaker, candidate_vector in sorted(candidate_prototypes.items()):
        matrix[candidate_speaker] = {
            reference_speaker: _cosine(candidate_vector, reference_vector)
            for reference_speaker, reference_vector in sorted(reference_prototypes.items())
        }

    reference_best_candidate: dict[str, str] = {}
    for reference_speaker in sorted(reference_prototypes):
        ranked = sorted(
            (
                (candidate_speaker, matrix[candidate_speaker][reference_speaker])
                for candidate_speaker in candidate_prototypes
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        reference_best_candidate[reference_speaker] = ranked[0][0]

    best_matches: list[dict[str, Any]] = []
    for candidate_speaker in sorted(candidate_prototypes):
        ranked = sorted(
            matrix[candidate_speaker].items(),
            key=lambda item: item[1],
            reverse=True,
        )
        reference_speaker, score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else None
        best_matches.append(
            {
                "candidate_speaker": candidate_speaker,
                "reference_speaker": reference_speaker,
                "score": score,
                "runner_up_score": runner_up,
                "margin": (score - runner_up) if runner_up is not None else None,
                "mutual_best": reference_best_candidate.get(reference_speaker)
                == candidate_speaker,
            }
        )

    return {
        "embedding_model": EMBEDDING_MODEL,
        "reference": reference_meta,
        "candidate": candidate_meta,
        "similarity_matrix": matrix,
        "best_matches": best_matches,
        "automatic_naming_enabled": False,
        "note": (
            "Cross-recording scores are for validation only. Confirm the recurring person "
            "manually from the representative timestamps/text before selecting a production threshold."
        ),
    }


def main() -> None:
    args = parse_args()
    report = run_cross_recording(args.reference, args.candidate)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Cross-recording benchmark report: {args.output}", flush=True)
    print(rendered, flush=True)


if __name__ == "__main__":
    main()

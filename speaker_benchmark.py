from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import whisperx
from pyannote.audio import Inference, Model
from pyannote.core import Segment

EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
SAMPLE_RATE = 16000
TARGET_CHUNK_SECONDS = 8.0
MIN_CHUNK_SECONDS = 4.0
MAX_CHUNKS_PER_SPEAKER = 6
MIN_PIECE_SECONDS = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark speaker embeddings on an already-transcribed recording."
    )
    parser.add_argument("source", type=Path, help="Source recording path")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser.parse_args()


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Embedding has zero/invalid norm.")
    return vector / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _split_long_interval(start: float, end: float) -> list[tuple[float, float]]:
    pieces: list[tuple[float, float]] = []
    cursor = start
    while end - cursor > TARGET_CHUNK_SECONDS:
        pieces.append((cursor, cursor + TARGET_CHUNK_SECONDS))
        cursor += TARGET_CHUNK_SECONDS
    if end - cursor >= MIN_PIECE_SECONDS:
        pieces.append((cursor, end))
    return pieces


def _build_chunks(segments: list[dict[str, Any]]) -> dict[str, list[list[tuple[float, float]]]]:
    by_speaker: dict[str, list[tuple[float, float]]] = {}
    for segment in segments:
        speaker = segment.get("speaker") or "UNKNOWN"
        if speaker == "UNKNOWN":
            continue
        start = segment.get("start")
        end = segment.get("end")
        if start is None or end is None:
            continue
        start_f, end_f = float(start), float(end)
        if end_f - start_f < MIN_PIECE_SECONDS:
            continue
        by_speaker.setdefault(speaker, []).extend(_split_long_interval(start_f, end_f))

    result: dict[str, list[list[tuple[float, float]]]] = {}
    for speaker, pieces in sorted(by_speaker.items()):
        pieces.sort()
        chunks: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = []
        duration = 0.0

        for piece in pieces:
            current.append(piece)
            duration += piece[1] - piece[0]
            if duration >= TARGET_CHUNK_SECONDS:
                chunks.append(current)
                current = []
                duration = 0.0
                if len(chunks) >= MAX_CHUNKS_PER_SPEAKER:
                    break

        if current and duration >= MIN_CHUNK_SECONDS and len(chunks) < MAX_CHUNKS_PER_SPEAKER:
            chunks.append(current)

        if len(chunks) >= 2:
            result[speaker] = chunks

    return result


def _extract_embeddings(
    source: Path,
    chunks: dict[str, list[list[tuple[float, float]]]],
) -> dict[str, list[np.ndarray]]:
    print(f"Loading audio for speaker benchmark: {source}", flush=True)
    audio = whisperx.load_audio(str(source))
    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    audio_file = {"waveform": waveform, "sample_rate": SAMPLE_RATE, "uri": source.stem}

    print(f"Loading speaker embedding model: {EMBEDDING_MODEL}", flush=True)
    model = Model.from_pretrained(EMBEDDING_MODEL)
    if model is None:
        raise RuntimeError(f"Could not load speaker embedding model: {EMBEDDING_MODEL}")

    inference = Inference(model, window="whole")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference.to(device)

    embeddings: dict[str, list[np.ndarray]] = {}
    try:
        for speaker, speaker_chunks in chunks.items():
            embeddings[speaker] = []
            for index, chunk in enumerate(speaker_chunks, start=1):
                segments = [Segment(start, end) for start, end in chunk]
                total = sum(end - start for start, end in chunk)
                print(
                    f"Embedding {speaker} chunk {index}/{len(speaker_chunks)} "
                    f"({total:.1f}s speech)...",
                    flush=True,
                )
                vector = inference.crop(audio_file, segments)
                embeddings[speaker].append(_normalize(np.asarray(vector)))
    finally:
        del inference, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return embeddings


def _pairwise_metrics(embeddings: dict[str, list[np.ndarray]]) -> tuple[list[float], list[float]]:
    same: list[float] = []
    different: list[float] = []
    speakers = sorted(embeddings)

    for speaker in speakers:
        vectors = embeddings[speaker]
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                same.append(_cosine(vectors[i], vectors[j]))

    for i, left in enumerate(speakers):
        for right in speakers[i + 1 :]:
            for a in embeddings[left]:
                for b in embeddings[right]:
                    different.append(_cosine(a, b))

    return same, different


def _closed_set_holdout(embeddings: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    prototypes: dict[str, np.ndarray] = {}
    queries: list[tuple[str, np.ndarray]] = []

    for speaker, vectors in sorted(embeddings.items()):
        refs = vectors[::2]
        held_out = vectors[1::2]
        if not refs or not held_out:
            continue
        prototypes[speaker] = _normalize(np.mean(np.stack(refs), axis=0))
        queries.extend((speaker, vector) for vector in held_out)

    details: list[dict[str, Any]] = []
    correct = 0
    for expected, query in queries:
        scores = sorted(
            ((speaker, _cosine(query, prototype)) for speaker, prototype in prototypes.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        predicted, top_score = scores[0]
        runner_up = scores[1][1] if len(scores) > 1 else None
        is_correct = predicted == expected
        correct += int(is_correct)
        details.append(
            {
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "top_score": top_score,
                "runner_up_score": runner_up,
                "margin": (top_score - runner_up) if runner_up is not None else None,
            }
        )

    return {
        "queries": len(queries),
        "correct": correct,
        "accuracy": (correct / len(queries)) if queries else None,
        "details": details,
    }


def run_benchmark(source: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source recording not found: {source}")

    transcript_path = source.with_name(f"{source.stem}_transcript.json")
    if not transcript_path.is_file():
        raise FileNotFoundError(
            f"Canonical transcript not found: {transcript_path}. Transcribe the recording first."
        )

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    chunks = _build_chunks(transcript.get("segments") or [])
    if len(chunks) < 2:
        raise RuntimeError(
            "Need at least two speakers with enough clean speech for the embedding benchmark."
        )

    embeddings = _extract_embeddings(source, chunks)
    same_scores, different_scores = _pairwise_metrics(embeddings)
    same = _stats(same_scores)
    different = _stats(different_scores)
    holdout = _closed_set_holdout(embeddings)

    same_p10 = same.get("p10")
    different_p95 = different.get("p95")
    separation_gap = None
    candidate_threshold = None
    if isinstance(same_p10, float) and isinstance(different_p95, float):
        separation_gap = same_p10 - different_p95
        if separation_gap > 0:
            candidate_threshold = (same_p10 + different_p95) / 2.0

    return {
        "source": str(source),
        "transcript": str(transcript_path),
        "embedding_model": EMBEDDING_MODEL,
        "speaker_count": len(embeddings),
        "chunks_per_speaker": {speaker: len(vectors) for speaker, vectors in embeddings.items()},
        "same_speaker_similarity": same,
        "different_speaker_similarity": different,
        "separation_gap_p10_vs_p95": separation_gap,
        "closed_set_holdout": holdout,
        "calibration": {
            "candidate_threshold": candidate_threshold,
            "ready_for_auto_assignment": False,
            "note": (
                "This single-recording benchmark can test embedding stability/separation, "
                "but automatic naming must remain disabled until the threshold is validated "
                "on a second recording containing at least one known speaker."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    report = run_benchmark(args.source)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Benchmark report: {args.output}", flush=True)
    print(rendered, flush=True)


if __name__ == "__main__":
    main()

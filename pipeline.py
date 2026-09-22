from __future__ import annotations

import gc
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
WHISPER_MODEL = "large-v3"
LANGUAGE = "ru"
BATCH_SIZE = 16
PROGRESS_PREFIX = "TRANSCRIBER_PROGRESS"


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def emit_progress(percent: float, stage: str, detail: str = "") -> None:
    """Emit a machine-readable progress line for the Gradio wrapper."""
    bounded = max(0.0, min(100.0, float(percent)))
    clean_stage = stage.replace("|", "/").replace("\n", " ").strip()
    clean_detail = detail.replace("|", "/").replace("\n", " ").strip()
    print(
        f"{PROGRESS_PREFIX}|{bounded:.1f}|{clean_stage}|{clean_detail}",
        flush=True,
    )


def _mapped_progress(
    start_percent: float,
    end_percent: float,
    stage: str,
) -> Callable[[float], None]:
    """Map WhisperX stage-local 0..100 progress into overall pipeline progress."""
    last_emitted = [-1]

    def callback(stage_percent: float) -> None:
        local = max(0.0, min(100.0, float(stage_percent)))
        overall = start_percent + (end_percent - start_percent) * (local / 100.0)
        rounded = int(overall)
        if rounded <= last_emitted[0]:
            return
        last_emitted[0] = rounded
        emit_progress(
            rounded,
            stage,
            f"{local:.0f}% of this stage",
        )

    return callback


def transcribe_recording(source: Path, hf_token: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the MVP baseline. Select a Colab T4 runtime.")

    device = "cuda"
    emit_progress(1, "Preparing audio")
    print(f"Loading audio: {source}", flush=True)
    audio = whisperx.load_audio(str(source))
    emit_progress(4, "Audio loaded")

    emit_progress(6, f"Loading Whisper {WHISPER_MODEL}")
    print(f"Transcribing with {WHISPER_MODEL} ({LANGUAGE})...", flush=True)
    model = whisperx.load_model(
        WHISPER_MODEL,
        device,
        compute_type="float16",
        language=LANGUAGE,
    )
    emit_progress(10, "Transcribing speech", "Whisper model loaded")
    result = model.transcribe(
        audio,
        batch_size=BATCH_SIZE,
        language=LANGUAGE,
        progress_callback=_mapped_progress(10, 62, "Transcribing speech"),
    )
    del model
    _release_cuda()
    emit_progress(62, "Transcription complete")

    emit_progress(64, "Loading alignment model")
    print("Aligning word timestamps...", flush=True)
    align_model, metadata = whisperx.load_align_model(
        language_code=result.get("language", LANGUAGE),
        device=device,
    )
    result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
        progress_callback=_mapped_progress(65, 80, "Aligning word timestamps"),
    )
    del align_model
    _release_cuda()
    emit_progress(80, "Alignment complete")

    emit_progress(82, "Loading diarization model")
    print(f"Diarizing with {DIARIZATION_MODEL}...", flush=True)
    diarizer = DiarizationPipeline(
        model_name=DIARIZATION_MODEL,
        token=hf_token,
        device=device,
    )
    diarization = diarizer(
        audio,
        progress_callback=_mapped_progress(84, 94, "Diarizing speakers"),
    )
    emit_progress(95, "Assigning speakers to transcript")
    result = whisperx.assign_word_speakers(diarization, result)
    del diarizer, diarization
    _release_cuda()

    speakers = sorted(
        {
            speaker
            for segment in result.get("segments", [])
            if (speaker := segment.get("speaker")) and speaker != "UNKNOWN"
        }
    )

    return {
        "source_path": str(source),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "language": result.get("language", LANGUAGE),
        "models": {
            "whisper": WHISPER_MODEL,
            "diarization": DIARIZATION_MODEL,
            "packages": {
                "whisperx": _package_version("whisperx"),
                "torch": _package_version("torch"),
                "pyannote.audio": _package_version("pyannote.audio"),
                "transformers": _package_version("transformers"),
            },
        },
        "detected_speakers": speakers,
        "speaker_count": len(speakers),
        "segments": result.get("segments", []),
    }

from __future__ import annotations

import gc
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
WHISPER_MODEL = "large-v3"
LANGUAGE = "ru"
BATCH_SIZE = 16


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def transcribe_recording(source: Path, hf_token: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the MVP baseline. Select a Colab T4 runtime.")

    device = "cuda"
    print(f"Loading audio: {source}")
    audio = whisperx.load_audio(str(source))

    print(f"Transcribing with {WHISPER_MODEL} ({LANGUAGE})...")
    model = whisperx.load_model(
        WHISPER_MODEL,
        device,
        compute_type="float16",
        language=LANGUAGE,
    )
    result = model.transcribe(audio, batch_size=BATCH_SIZE, language=LANGUAGE)
    del model
    _release_cuda()

    print("Aligning word timestamps...")
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
    )
    del align_model
    _release_cuda()

    print(f"Diarizing with {DIARIZATION_MODEL}...")
    diarizer = DiarizationPipeline(
        model_name=DIARIZATION_MODEL,
        token=hf_token,
        device=device,
    )
    diarization = diarizer(audio)
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

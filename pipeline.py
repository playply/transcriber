from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
WHISPER_MODEL = "large-v3"
LANGUAGE = "ru"


def transcribe_recording(source: Path, hf_token: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the MVP baseline. Select a Colab T4 runtime.")

    device = "cuda"
    audio = whisperx.load_audio(str(source))

    model = whisperx.load_model(
        WHISPER_MODEL,
        device,
        compute_type="float16",
        language=LANGUAGE,
    )
    result = model.transcribe(audio, batch_size=16, language=LANGUAGE)

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

    diarizer = DiarizationPipeline(
        model_name=DIARIZATION_MODEL,
        token=hf_token,
        device=device,
    )
    diarization = diarizer(audio)
    result = whisperx.assign_word_speakers(diarization, result)

    speakers = sorted(
        {
            segment.get("speaker")
            for segment in result.get("segments", [])
            if segment.get("speaker")
        }
    )

    return {
        "source_path": str(source),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "language": result.get("language", LANGUAGE),
        "models": {
            "whisper": WHISPER_MODEL,
            "diarization": DIARIZATION_MODEL,
        },
        "detected_speakers": speakers,
        "speaker_count": len(speakers),
        "segments": result.get("segments", []),
    }

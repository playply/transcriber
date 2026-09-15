from __future__ import annotations

import argparse
import os
from pathlib import Path

from exporters import export_transcript
from pipeline import transcribe_recording

SUPPORTED_EXTENSIONS = {".mp4", ".mp3", ".m4a", ".wav"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interview Transcriber clean-room baseline")
    parser.add_argument("source", type=Path, help="Path to a recording on mounted Google Drive")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Source recording not found: {source}")
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format {source.suffix!r}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is missing. Load it from Colab Secrets before running app.py.")

    result = transcribe_recording(source, hf_token=hf_token)
    outputs = export_transcript(result, source)

    print("Transcription complete.")
    for label, path in outputs.items():
        print(f"{label.upper()}: {path}")


if __name__ == "__main__":
    main()

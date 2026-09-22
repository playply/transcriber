from __future__ import annotations

import argparse
import os
from pathlib import Path

from exporters import export_transcript
from pipeline import emit_progress, transcribe_recording
from speaker_registry import load_registry
from speaker_resolution import apply_recognition

SUPPORTED_EXTENSIONS = {".mp4", ".mp3", ".m4a", ".wav"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interview Transcriber MVP")
    parser.add_argument("source", type=Path, help="Path to a recording on mounted Google Drive")
    return parser.parse_args()


def _registry_path() -> Path:
    configured = os.environ.get("TRANSCRIBER_REGISTRY_PATH")
    if configured:
        return Path(configured).expanduser().resolve()

    drive_root = Path(
        os.environ.get("TRANSCRIBER_DRIVE_ROOT", "/content/drive/MyDrive")
    ).expanduser().resolve()
    return drive_root / "Interview Transcriber" / "_speaker_registry" / "registry.json"


def _apply_registry_if_available(source: Path) -> dict | None:
    registry_path = _registry_path()
    if not registry_path.is_file():
        print("Speaker registry is empty; leaving diarized speakers unresolved.", flush=True)
        return None

    try:
        registry = load_registry(registry_path)
        identities = [
            identity
            for identity in registry.get("speakers") or []
            if identity.get("embeddings")
        ]
        if not identities:
            print("Speaker registry has no usable identities; leaving speakers unresolved.", flush=True)
            return None

        print(
            f"Recognizing speakers against registry ({len(identities)} known identities)...",
            flush=True,
        )
        report = apply_recognition(source, registry_path)
        known = [
            item
            for item in report.get("results") or []
            if item.get("status") == "KNOWN" and item.get("resolved_name")
        ]
        if known:
            rendered = ", ".join(
                f"{item['diarization_speaker']}={item['resolved_name']} "
                f"(score={float(item['score']):.3f})"
                for item in known
            )
            print(f"✓ Known speakers resolved: {rendered}", flush=True)
        else:
            print("✓ Registry checked; no speaker met the confidence rules.", flush=True)
        return report
    except Exception as exc:
        # The core transcription has already been saved. A recognition failure must not
        # destroy a successful transcript; leave names unresolved and make the failure visible.
        print(
            "WARNING: speaker recognition failed after transcription; "
            f"outputs remain available with unresolved speaker IDs. {type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


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

    emit_progress(96, "Saving transcript files")
    outputs = export_transcript(result, source)

    emit_progress(98, "Recognizing known speakers")
    _apply_registry_if_available(source)

    emit_progress(100, "Complete", "Transcript and speaker recognition finished")
    print("Transcription complete.", flush=True)
    for label, path in outputs.items():
        print(f"{label.upper()}: {path}", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Iterator

import gradio as gr

DRIVE_ROOT = Path(os.environ.get("TRANSCRIBER_DRIVE_ROOT", "/content/drive/MyDrive")).resolve()
APP_PATH = Path(__file__).with_name("app.py").resolve()
SUPPORTED_EXTENSIONS = {".mp4", ".mp3", ".m4a", ".wav"}


def _resolve_source(selected: str | None) -> Path:
    if not selected:
        raise gr.Error("Choose a recording from Google Drive first.")

    source = Path(selected)
    if not source.is_absolute():
        source = DRIVE_ROOT / source
    source = source.resolve()

    try:
        source.relative_to(DRIVE_ROOT)
    except ValueError as exc:
        raise gr.Error("The selected file must be inside the mounted Google Drive.") from exc

    if not source.is_file():
        raise gr.Error(f"File not found: {source}")
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise gr.Error(f"Unsupported format {source.suffix}. Supported: {supported}")

    return source


def _output_paths(source: Path) -> tuple[Path, Path, Path]:
    stem = source.stem
    return (
        source.with_name(f"{stem}_transcript.json"),
        source.with_name(f"{stem}_transcript.txt"),
        source.with_name(f"{stem}_transcript.docx"),
    )


def _result_summary(json_path: Path) -> str:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return "Transcription complete."

    speaker_count = data.get("speaker_count")
    speakers = data.get("detected_speakers") or []
    segment_count = len(data.get("segments") or [])

    bits = ["Transcription complete."]
    if speaker_count is not None:
        bits.append(f"Detected speakers: {speaker_count}.")
    if speakers:
        bits.append("IDs: " + ", ".join(map(str, speakers)) + ".")
    if segment_count:
        bits.append(f"Segments: {segment_count}.")
    bits.append("JSON, TXT and DOCX were saved beside the source recording in Google Drive.")
    return " ".join(bits)


def transcribe(selected: str | None) -> Iterator[tuple[str, str | None, str | None, str | None]]:
    source = _resolve_source(selected)

    if not os.environ.get("HF_TOKEN"):
        raise gr.Error("HF_TOKEN is not available. Restart the launcher and check Colab Secrets.")

    yield (
        "Processing. The first run can take longer while WhisperX/pyannote models are downloaded...",
        None,
        None,
        None,
    )

    env = os.environ.copy()
    process = subprocess.Popen(
        [sys.executable, str(APP_PATH), str(source)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    last_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        last_lines.append(line.rstrip())
        if len(last_lines) > 80:
            del last_lines[:-80]

    return_code = process.wait()
    if return_code != 0:
        tail = "\n".join(last_lines[-20:])
        raise gr.Error(
            "Transcription failed. The last log lines are shown below:\n\n" + tail
        )

    json_path, txt_path, docx_path = _output_paths(source)
    missing = [str(path) for path in (json_path, txt_path, docx_path) if not path.exists()]
    if missing:
        raise gr.Error("Pipeline finished but expected output files are missing: " + ", ".join(missing))

    yield (
        _result_summary(json_path),
        str(json_path),
        str(txt_path),
        str(docx_path),
    )


def build_ui() -> gr.Blocks:
    if not DRIVE_ROOT.exists():
        raise RuntimeError(f"Google Drive is not mounted: {DRIVE_ROOT}")

    with gr.Blocks(title="Interview Transcriber") as demo:
        gr.Markdown(
            "# Interview Transcriber\n"
            "Choose one existing recording from Google Drive and start transcription. "
            "Outputs are saved next to the source file."
        )
        recording = gr.FileExplorer(
            root_dir=str(DRIVE_ROOT),
            glob="**/*",
            file_count="single",
            label="Recording on Google Drive",
            height=420,
        )
        start = gr.Button("Transcribe", variant="primary")
        status = gr.Textbox(label="Status", interactive=False, lines=4)
        with gr.Row():
            json_output = gr.File(label="JSON")
            txt_output = gr.File(label="TXT")
            docx_output = gr.File(label="DOCX")

        start.click(
            fn=transcribe,
            inputs=recording,
            outputs=[status, json_output, txt_output, docx_output],
            concurrency_limit=1,
        )

    return demo


def main() -> None:
    username = "transcriber"
    password = os.environ.get("TRANSCRIBER_UI_PASSWORD") or secrets.token_urlsafe(12)

    print("\nInterview Transcriber UI credentials")
    print(f"Username: {username}")
    print(f"Password: {password}")
    print("Keep the temporary Gradio URL and password private.\n")

    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        share=True,
        auth=(username, password),
        allowed_paths=[str(DRIVE_ROOT)],
        show_error=True,
    )


if __name__ == "__main__":
    main()

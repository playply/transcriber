from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator

print("Loading Gradio UI...", flush=True)
import gradio as gr

DRIVE_ROOT = Path(os.environ.get("TRANSCRIBER_DRIVE_ROOT", "/content/drive/MyDrive")).resolve()
APP_PATH = Path(__file__).with_name("app.py").resolve()
BENCHMARK_PATH = Path(__file__).with_name("speaker_benchmark.py").resolve()
CROSS_BENCHMARK_PATH = Path(__file__).with_name("speaker_cross_benchmark.py").resolve()
APP_PYTHON = os.environ.get("TRANSCRIBER_APP_PYTHON") or sys.executable
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
        [APP_PYTHON, "-u", str(APP_PATH), str(source)],
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


def _fmt_score(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _format_time(seconds: object) -> str:
    if seconds is None:
        return "n/a"
    total = max(0, int(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _benchmark_summary(report: dict) -> str:
    same = report.get("same_speaker_similarity") or {}
    different = report.get("different_speaker_similarity") or {}
    holdout = report.get("closed_set_holdout") or {}
    calibration = report.get("calibration") or {}
    chunks = report.get("chunks_per_speaker") or {}

    accuracy = holdout.get("accuracy")
    accuracy_text = "n/a" if accuracy is None else f"{float(accuracy) * 100:.1f}%"
    threshold = calibration.get("candidate_threshold")
    threshold_text = _fmt_score(threshold)

    return (
        f"Embedding model: {report.get('embedding_model')}\n"
        f"Chunks per speaker: {chunks}\n"
        f"Same-speaker similarity: p10={_fmt_score(same.get('p10'))}, "
        f"median={_fmt_score(same.get('median'))}\n"
        f"Different-speaker similarity: p95={_fmt_score(different.get('p95'))}, "
        f"max={_fmt_score(different.get('max'))}\n"
        f"Conservative separation gap (same p10 - different p95): "
        f"{_fmt_score(report.get('separation_gap_p10_vs_p95'))}\n"
        f"Closed-set holdout accuracy: {accuracy_text} "
        f"({holdout.get('correct', 0)}/{holdout.get('queries', 0)})\n"
        f"Exploratory threshold: {threshold_text}\n\n"
        "Automatic naming is still disabled. We will only standardize a threshold after "
        "a second-recording validation with at least one recurring speaker."
    )


def benchmark_speakers(selected: str | None) -> Iterator[str]:
    source = _resolve_source(selected)
    transcript_path = source.with_name(f"{source.stem}_transcript.json")
    if not transcript_path.is_file():
        raise gr.Error("Transcribe this recording first; the canonical JSON is required.")

    yield (
        "Running speaker embedding benchmark. This uses the existing transcript to build "
        "multiple speech chunks per diarized speaker and does not change the transcript or registry."
    )

    env = os.environ.copy()
    with tempfile.NamedTemporaryFile(prefix="speaker-benchmark-", suffix=".json", delete=False) as tmp:
        report_path = Path(tmp.name)

    process = subprocess.Popen(
        [
            APP_PYTHON,
            "-u",
            str(BENCHMARK_PATH),
            str(source),
            "--output",
            str(report_path),
        ],
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
        report_path.unlink(missing_ok=True)
        tail = "\n".join(last_lines[-20:])
        raise gr.Error("Speaker benchmark failed. Last log lines:\n\n" + tail)

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    finally:
        report_path.unlink(missing_ok=True)

    yield _benchmark_summary(report)


def _speaker_examples(meta: dict, label: str) -> str:
    chunks = meta.get("chunks_per_speaker") or {}
    examples = meta.get("representative_segments") or {}
    lines = [f"{label} speaker examples:"]
    for speaker in sorted(chunks):
        example = examples.get(speaker) or {}
        text = str(example.get("text") or "").replace("\n", " ").strip()
        if len(text) > 140:
            text = text[:137] + "..."
        lines.append(
            f"  {speaker}: {_format_time(example.get('start'))} "
            f"({chunks[speaker]} chunks) — {text or '[no text]'}"
        )
    return "\n".join(lines)


def _cross_benchmark_summary(report: dict) -> str:
    reference = report.get("reference") or {}
    candidate = report.get("candidate") or {}
    matrix = report.get("similarity_matrix") or {}
    matches = report.get("best_matches") or []

    reference_speakers = sorted((reference.get("chunks_per_speaker") or {}).keys())
    lines = [
        f"Embedding model: {report.get('embedding_model')}",
        f"Reference: {Path(str(reference.get('source', ''))).name}",
        f"Second recording: {Path(str(candidate.get('source', ''))).name}",
        "",
        _speaker_examples(reference, "Reference"),
        "",
        _speaker_examples(candidate, "Second recording"),
        "",
        "Cross-recording cosine similarity matrix:",
    ]

    if reference_speakers and matrix:
        header = "candidate \\ reference | " + " | ".join(reference_speakers)
        lines.append(header)
        lines.append("-" * len(header))
        for candidate_speaker in sorted(matrix):
            row = matrix[candidate_speaker]
            scores = " | ".join(_fmt_score(row.get(ref)) for ref in reference_speakers)
            lines.append(f"{candidate_speaker} | {scores}")

    lines.extend(["", "Best matches (validation only):"])
    for match in matches:
        margin = match.get("margin")
        mutual = "yes" if match.get("mutual_best") else "no"
        lines.append(
            f"  {match.get('candidate_speaker')} -> {match.get('reference_speaker')}: "
            f"score={_fmt_score(match.get('score'))}, "
            f"margin={_fmt_score(margin)}, mutual_best={mutual}"
        )

    lines.extend(
        [
            "",
            "Automatic naming is still disabled. Confirm which pair is actually the same person "
            "from the timestamps/text above; the score is not yet a production threshold.",
        ]
    )
    return "\n".join(lines)


def compare_recordings(
    reference_selected: str | None,
    candidate_selected: str | None,
) -> Iterator[str]:
    reference = _resolve_source(reference_selected)
    candidate = _resolve_source(candidate_selected)
    if reference == candidate:
        raise gr.Error("Choose two different recordings.")

    for source in (reference, candidate):
        transcript_path = source.with_name(f"{source.stem}_transcript.json")
        if not transcript_path.is_file():
            raise gr.Error(
                f"Transcribe {source.name} first; its canonical JSON is required."
            )

    yield (
        "Comparing speaker embeddings across two recordings. This is validation-only: "
        "no names are assigned and nothing is written to the speaker registry."
    )

    env = os.environ.copy()
    with tempfile.NamedTemporaryFile(prefix="speaker-cross-", suffix=".json", delete=False) as tmp:
        report_path = Path(tmp.name)

    process = subprocess.Popen(
        [
            APP_PYTHON,
            "-u",
            str(CROSS_BENCHMARK_PATH),
            str(reference),
            str(candidate),
            "--output",
            str(report_path),
        ],
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
        report_path.unlink(missing_ok=True)
        tail = "\n".join(last_lines[-20:])
        raise gr.Error("Cross-recording benchmark failed. Last log lines:\n\n" + tail)

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    finally:
        report_path.unlink(missing_ok=True)

    yield _cross_benchmark_summary(report)


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

        with gr.Accordion("Speaker recognition benchmark (development)", open=False):
            gr.Markdown(
                "Runs non-destructive speaker-embedding validation. "
                "It does not name speakers or write to a speaker registry."
            )
            benchmark = gr.Button("Benchmark selected recording")
            benchmark_status = gr.Textbox(label="Single-recording benchmark", interactive=False, lines=10)
            benchmark.click(
                fn=benchmark_speakers,
                inputs=recording,
                outputs=benchmark_status,
                concurrency_limit=1,
            )

            gr.Markdown(
                "### Cross-recording validation\n"
                "Use the recording selected above as the reference. Choose a second recording "
                "that has already been transcribed and contains at least one recurring person."
            )
            comparison_recording = gr.FileExplorer(
                root_dir=str(DRIVE_ROOT),
                glob="**/*",
                file_count="single",
                label="Second transcribed recording on Google Drive",
                height=320,
            )
            compare = gr.Button("Compare speakers across recordings")
            compare_status = gr.Textbox(
                label="Cross-recording benchmark",
                interactive=False,
                lines=18,
            )
            compare.click(
                fn=compare_recordings,
                inputs=[recording, comparison_recording],
                outputs=compare_status,
                concurrency_limit=1,
            )

    return demo


def main() -> None:
    username = "transcriber"
    password = os.environ.get("TRANSCRIBER_UI_PASSWORD") or secrets.token_urlsafe(12)

    print("\nInterview Transcriber UI credentials", flush=True)
    print(f"Username: {username}", flush=True)
    print(f"Password: {password}", flush=True)
    print("Keep the temporary Gradio URL and password private.\n", flush=True)

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

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
REGISTRY_TOOL_PATH = Path(__file__).with_name("speaker_registry.py").resolve()
APP_PYTHON = os.environ.get("TRANSCRIBER_APP_PYTHON") or sys.executable
REGISTRY_PATH = Path(
    os.environ.get(
        "TRANSCRIBER_REGISTRY_PATH",
        str(DRIVE_ROOT / "Interview Transcriber" / "_speaker_registry" / "registry.json"),
    )
).resolve()
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
        "Automatic naming is still disabled in transcript outputs."
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
        [APP_PYTHON, "-u", str(BENCHMARK_PATH), str(source), "--output", str(report_path)],
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
            "Automatic naming in transcript outputs is still disabled. Use the registry section "
            "above to save confirmed identities and test recognition."
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
            raise gr.Error(f"Transcribe {source.name} first; its canonical JSON is required.")

    yield (
        "Comparing speaker embeddings across two recordings. This is validation-only: "
        "no names are assigned to transcript outputs."
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


def _transcript_speaker_info(source: Path) -> tuple[list[str], str]:
    transcript_path = source.with_name(f"{source.stem}_transcript.json")
    if not transcript_path.is_file():
        raise gr.Error("Transcribe this recording first; the canonical JSON is required.")

    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = data.get("segments") or []
    speakers = sorted(
        {
            str(segment.get("speaker"))
            for segment in segments
            if segment.get("speaker") and segment.get("speaker") != "UNKNOWN"
        }
    )
    lines: list[str] = []
    for speaker in speakers:
        candidates = [
            segment
            for segment in segments
            if segment.get("speaker") == speaker
            and segment.get("start") is not None
            and segment.get("end") is not None
        ]
        if candidates:
            example = max(
                candidates,
                key=lambda segment: float(segment["end"]) - float(segment["start"]),
            )
            text = str(example.get("text") or "").replace("\n", " ").strip()
            if len(text) > 150:
                text = text[:147] + "..."
            lines.append(
                f"{speaker}: {_format_time(example.get('start'))} — {text or '[no text]'}"
            )
        else:
            lines.append(f"{speaker}: no representative segment")
    return speakers, "\n".join(lines)


def load_registry_speakers(selected: str | None):
    source = _resolve_source(selected)
    speakers, preview = _transcript_speaker_info(source)
    value = speakers[0] if speakers else None
    return gr.Dropdown(choices=speakers, value=value), preview


def _run_registry_tool(arguments: list[str], error_prefix: str) -> dict:
    env = os.environ.copy()
    with tempfile.NamedTemporaryFile(prefix="speaker-registry-", suffix=".json", delete=False) as tmp:
        report_path = Path(tmp.name)

    command = [
        APP_PYTHON,
        "-u",
        str(REGISTRY_TOOL_PATH),
        *arguments,
        "--output",
        str(report_path),
    ]
    process = subprocess.Popen(
        command,
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
        raise gr.Error(f"{error_prefix}. Last log lines:\n\n{tail}")

    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    finally:
        report_path.unlink(missing_ok=True)


def enroll_registry_speaker(
    selected: str | None,
    diarization_speaker: str | None,
    display_name: str | None,
) -> Iterator[str]:
    source = _resolve_source(selected)
    if not diarization_speaker:
        raise gr.Error("Load speakers and choose the speaker to save.")
    name = (display_name or "").strip()
    if not name:
        raise gr.Error("Enter a name for this speaker.")

    yield f"Saving {diarization_speaker} as {name}. Computing reference embeddings..."
    report = _run_registry_tool(
        [
            "enroll",
            str(source),
            diarization_speaker,
            name,
            "--registry",
            str(REGISTRY_PATH),
        ],
        "Could not save speaker to registry",
    )
    action = "Created" if report.get("created") else "Updated"
    yield (
        f"{action} identity: {report.get('display_name')} ({report.get('speaker_id')})\n"
        f"Source speaker: {report.get('diarization_speaker')}\n"
        f"Reference chunks added: {report.get('chunks_added')}\n"
        f"Stored reference embeddings: {report.get('reference_embeddings')}\n"
        f"Registry: {report.get('registry')}\n\n"
        "Transcript outputs were not modified."
    )


def _recognition_summary(report: dict) -> str:
    thresholds = report.get("thresholds") or {}
    lines = [
        f"Registry: {report.get('registry')}",
        f"Known identities: {report.get('known_identity_count')}",
        "Thresholds: "
        f"similarity>={thresholds.get('similarity')}, "
        f"margin>={thresholds.get('margin')}, "
        f"min_chunks={thresholds.get('min_chunks')}",
        "",
    ]
    for item in report.get("results") or []:
        speaker = item.get("diarization_speaker")
        status = item.get("status")
        example = item.get("representative_segment") or {}
        snippet = str(example.get("text") or "").replace("\n", " ").strip()
        if len(snippet) > 110:
            snippet = snippet[:107] + "..."
        if status == "KNOWN":
            lines.append(
                f"{speaker} -> {item.get('resolved_name')} [KNOWN] "
                f"score={_fmt_score(item.get('score'))}, "
                f"margin={_fmt_score(item.get('margin'))}, "
                f"unique_best={'yes' if item.get('unique_best') else 'no'}, "
                f"chunks={item.get('chunks')}"
            )
        else:
            best = item.get("matched_display_name")
            best_text = f", best={best}" if best else ""
            lines.append(
                f"{speaker} -> UNKNOWN "
                f"(score={_fmt_score(item.get('score'))}{best_text}, "
                f"reason={item.get('reason')}, chunks={item.get('chunks')})"
            )
        if example:
            lines.append(
                f"  {_format_time(example.get('start'))} — {snippet or '[no text]'}"
            )
    lines.extend(
        [
            "",
            "Recognition is preview-only in this slice; transcript JSON/TXT/DOCX are not rewritten yet.",
        ]
    )
    return "\n".join(lines)


def recognize_registry_speakers(selected: str | None) -> Iterator[str]:
    source = _resolve_source(selected)
    if not REGISTRY_PATH.is_file():
        raise gr.Error("Speaker registry is empty. Save one confirmed speaker first.")

    yield "Comparing detected speakers with the persistent registry..."
    report = _run_registry_tool(
        ["recognize", str(source), "--registry", str(REGISTRY_PATH)],
        "Speaker recognition failed",
    )
    yield _recognition_summary(report)


def registry_overview() -> str:
    if not REGISTRY_PATH.is_file():
        return f"Registry is empty. It will be created at:\n{REGISTRY_PATH}"
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    identities = data.get("speakers") or []
    lines = [f"Registry: {REGISTRY_PATH}", f"Known identities: {len(identities)}"]
    for identity in identities:
        lines.append(
            f"  {identity.get('display_name')} ({identity.get('speaker_id')}): "
            f"{len(identity.get('embeddings') or [])} reference embeddings, "
            f"{len(identity.get('provenance') or [])} source(s)"
        )
    return "\n".join(lines)


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

        with gr.Accordion("Speaker registry (MVP)", open=True):
            gr.Markdown(
                "Save a confirmed diarized speaker once, then test recognition on another "
                "already-transcribed recording. Conservative matching: similarity >= 0.60, "
                "margin >= 0.20 when a runner-up exists, at least 2 usable chunks, and a unique best match."
            )
            refresh_speakers = gr.Button("Load speakers from selected recording")
            registry_speaker = gr.Dropdown(
                choices=[],
                label="Confirmed speaker in selected recording",
            )
            speaker_preview = gr.Textbox(
                label="Speaker examples",
                interactive=False,
                lines=6,
            )
            refresh_speakers.click(
                fn=load_registry_speakers,
                inputs=recording,
                outputs=[registry_speaker, speaker_preview],
            )

            display_name = gr.Textbox(
                label="Name for confirmed speaker",
                placeholder="e.g. Alex",
            )
            save_speaker = gr.Button("Save confirmed speaker to registry", variant="primary")
            registry_status = gr.Textbox(
                label="Registry status",
                interactive=False,
                lines=7,
            )
            save_speaker.click(
                fn=enroll_registry_speaker,
                inputs=[recording, registry_speaker, display_name],
                outputs=registry_status,
                concurrency_limit=1,
            )

            with gr.Row():
                recognize = gr.Button("Recognize known speakers in selected recording")
                show_registry = gr.Button("Show registry")
            recognition_status = gr.Textbox(
                label="Recognition preview",
                interactive=False,
                lines=14,
            )
            recognize.click(
                fn=recognize_registry_speakers,
                inputs=recording,
                outputs=recognition_status,
                concurrency_limit=1,
            )
            show_registry.click(
                fn=registry_overview,
                outputs=recognition_status,
            )

        with gr.Accordion("Speaker recognition benchmark (development)", open=False):
            gr.Markdown(
                "Runs non-destructive speaker-embedding validation. "
                "It does not modify transcript outputs."
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

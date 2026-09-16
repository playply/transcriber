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

DRIVE_ROOT = Path(
    os.environ.get("TRANSCRIBER_DRIVE_ROOT", "/content/drive/MyDrive")
).resolve()
APP_PATH = Path(__file__).with_name("app.py").resolve()
BENCHMARK_PATH = Path(__file__).with_name("speaker_benchmark.py").resolve()
CROSS_BENCHMARK_PATH = Path(__file__).with_name("speaker_cross_benchmark.py").resolve()
REGISTRY_TOOL_PATH = Path(__file__).with_name("speaker_registry.py").resolve()
CONFIRM_TOOL_PATH = Path(__file__).with_name("confirm_speaker.py").resolve()
APP_PYTHON = os.environ.get("TRANSCRIBER_APP_PYTHON") or sys.executable
GPU_AVAILABLE = os.environ.get("TRANSCRIBER_GPU_AVAILABLE", "1") == "1"
REPO_HEAD = os.environ.get("TRANSCRIBER_REPO_HEAD", "unknown")
REGISTRY_PATH = Path(
    os.environ.get(
        "TRANSCRIBER_REGISTRY_PATH",
        str(
            DRIVE_ROOT
            / "Interview Transcriber"
            / "_speaker_registry"
            / "registry.json"
        ),
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
        raise gr.Error(
            "The selected file must be inside the mounted Google Drive."
        ) from exc

    if not source.is_file():
        raise gr.Error(f"File not found: {source}")
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise gr.Error(
            f"Unsupported format {source.suffix}. Supported: {supported}"
        )
    return source


def _output_paths(source: Path) -> tuple[Path, Path, Path]:
    stem = source.stem
    return (
        source.with_name(f"{stem}_transcript.json"),
        source.with_name(f"{stem}_transcript.txt"),
        source.with_name(f"{stem}_transcript.docx"),
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


def _run_process(command: list[str], error_prefix: str) -> None:
    process = subprocess.Popen(
        command,
        env=os.environ.copy(),
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
        raise gr.Error(f"{error_prefix}. Last log lines:\n\n{tail}")


def _run_json_tool(command: list[str], error_prefix: str) -> dict:
    with tempfile.NamedTemporaryFile(
        prefix="transcriber-tool-", suffix=".json", delete=False
    ) as tmp:
        report_path = Path(tmp.name)

    try:
        _run_process(
            [*command, "--output", str(report_path)],
            error_prefix,
        )
        return json.loads(report_path.read_text(encoding="utf-8"))
    finally:
        report_path.unlink(missing_ok=True)


def _load_canonical(source: Path) -> dict:
    json_path = source.with_name(f"{source.stem}_transcript.json")
    if not json_path.is_file():
        raise gr.Error(
            "This recording has no canonical _transcript.json yet. Transcribe it when GPU is available."
        )
    return json.loads(json_path.read_text(encoding="utf-8"))


def _representative_segment(
    segments: list[dict],
    speaker: str,
) -> dict:
    candidates = [
        segment
        for segment in segments
        if str(segment.get("speaker") or "UNKNOWN") == speaker
        and segment.get("start") is not None
        and segment.get("end") is not None
    ]
    if not candidates:
        return {}
    return max(
        candidates,
        key=lambda segment: float(segment["end"]) - float(segment["start"]),
    )


def _unknown_speakers_from_data(
    data: dict,
) -> tuple[list[str], dict[str, str]]:
    segments = data.get("segments") or []
    concrete = sorted(
        {
            str(segment.get("speaker"))
            for segment in segments
            if segment.get("speaker")
            and segment.get("speaker") != "UNKNOWN"
        }
    )
    resolved = {
        str(segment.get("speaker"))
        for segment in segments
        if segment.get("speaker")
        and segment.get("speaker") != "UNKNOWN"
        and str(segment.get("resolved_name") or "").strip()
    }
    unknowns = [speaker for speaker in concrete if speaker not in resolved]

    resolution_results = {
        str(item.get("diarization_speaker")): item
        for item in (data.get("speaker_resolution") or {}).get("results") or []
    }
    unknowns.sort(
        key=lambda speaker: (
            -int((resolution_results.get(speaker) or {}).get("chunks") or 0),
            speaker,
        )
    )

    previews: dict[str, str] = {}
    for speaker in unknowns:
        example = _representative_segment(segments, speaker)
        text = str(example.get("text") or "").replace("\n", " ").strip()
        if len(text) > 180:
            text = text[:177] + "..."
        resolution = resolution_results.get(speaker) or {}
        chunks = resolution.get("chunks")
        reason = resolution.get("reason")
        metadata: list[str] = []
        if chunks is not None:
            metadata.append(f"usable chunks: {chunks}")
        if reason:
            metadata.append(f"recognition: {reason}")
        suffix = f"\n{', '.join(metadata)}" if metadata else ""
        previews[speaker] = (
            f"{speaker} at {_format_time(example.get('start'))}: "
            f"{text or '[no text]'}{suffix}"
        )
    return unknowns, previews


def _unknown_update(source: Path):
    data = _load_canonical(source)
    unknowns, previews = _unknown_speakers_from_data(data)
    selected = unknowns[0] if unknowns else None
    preview = previews.get(selected, "") if selected else ""
    if unknowns:
        status = (
            f"Unknown speakers needing a name: {len(unknowns)} "
            f"({', '.join(unknowns)})."
        )
    else:
        status = "No unresolved diarization speakers need a name."
    return (
        gr.Dropdown(
            choices=unknowns,
            value=selected,
            label="Unknown speaker",
        ),
        preview,
        status,
    )


def load_existing_unknowns(selected: str | None):
    source = _resolve_source(selected)
    return _unknown_update(source)


def _result_summary(json_path: Path) -> str:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return "Transcription complete."

    bits = ["Transcription complete."]
    if data.get("speaker_count") is not None:
        bits.append(f"Detected speakers: {data['speaker_count']}.")
    speakers = data.get("detected_speakers") or []
    if speakers:
        bits.append("IDs: " + ", ".join(map(str, speakers)) + ".")
    segment_count = len(data.get("segments") or [])
    if segment_count:
        bits.append(f"Segments: {segment_count}.")
    unknowns, _ = _unknown_speakers_from_data(data)
    if unknowns:
        bits.append(
            "Unknown speakers needing confirmation: "
            + ", ".join(unknowns)
            + "."
        )
    else:
        bits.append("No unresolved diarization speaker IDs remain.")
    bits.append(
        "JSON, TXT and DOCX were saved beside the source recording in Google Drive."
    )
    return " ".join(bits)


def transcribe(selected: str | None) -> Iterator[tuple]:
    if not GPU_AVAILABLE:
        raise gr.Error(
            "GPU is unavailable in this Colab runtime. Use CPU maintenance mode for existing transcripts, "
            "or start a GPU runtime later to transcribe a new recording."
        )

    source = _resolve_source(selected)
    if not os.environ.get("HF_TOKEN"):
        raise gr.Error(
            "HF_TOKEN is not available. Restart the launcher and check Colab Secrets."
        )

    yield (
        "Processing. The first run can take longer while WhisperX/pyannote models are downloaded...",
        None,
        None,
        None,
        gr.Dropdown(choices=[], value=None, label="Unknown speaker"),
        "",
        "Unknown speakers will appear here after transcription.",
    )
    _run_process(
        [APP_PYTHON, "-u", str(APP_PATH), str(source)],
        "Transcription failed",
    )

    json_path, txt_path, docx_path = _output_paths(source)
    missing = [
        str(path)
        for path in (json_path, txt_path, docx_path)
        if not path.exists()
    ]
    if missing:
        raise gr.Error(
            "Pipeline finished but expected output files are missing: "
            + ", ".join(missing)
        )

    unknown_dropdown, unknown_preview, unknown_status = _unknown_update(source)
    yield (
        _result_summary(json_path),
        str(json_path),
        str(txt_path),
        str(docx_path),
        unknown_dropdown,
        unknown_preview,
        unknown_status,
    )


def unknown_speaker_preview(
    selected: str | None,
    diarization_speaker: str | None,
) -> str:
    source = _resolve_source(selected)
    if not diarization_speaker:
        return ""
    data = _load_canonical(source)
    unknowns, previews = _unknown_speakers_from_data(data)
    if diarization_speaker not in unknowns:
        return "This speaker is already resolved."
    return previews.get(diarization_speaker, "")


def confirm_unknown_speaker(
    selected: str | None,
    diarization_speaker: str | None,
    display_name: str | None,
):
    source = _resolve_source(selected)
    if not diarization_speaker:
        raise gr.Error("Choose an unknown speaker first.")
    name = (display_name or "").strip()
    if not name:
        raise gr.Error("Enter the speaker's name.")

    data = _load_canonical(source)
    unknowns, _ = _unknown_speakers_from_data(data)
    if diarization_speaker not in unknowns:
        raise gr.Error(
            f"{diarization_speaker} is no longer unresolved. Reload unknown speakers and choose another speaker."
        )

    report = _run_json_tool(
        [
            APP_PYTHON,
            "-u",
            str(CONFIRM_TOOL_PATH),
            str(source),
            diarization_speaker,
            name,
            "--registry",
            str(REGISTRY_PATH),
        ],
        "Could not confirm unknown speaker",
    )
    outputs = report.get("outputs") or {}
    unknown_dropdown, preview, remaining_status = _unknown_update(source)
    if report.get("voice_reference_saved", True):
        voice_status = (
            f"Voice reference saved: {report.get('chunks_added')} usable chunk(s)."
        )
    else:
        voice_status = (
            "Name saved, but no voice reference was added because this recording has too little usable speech. "
            "The identity will not be auto-matched until a later recording provides enough speech."
        )
    return (
        (
            f"Saved {diarization_speaker} as {report.get('display_name')} "
            f"({report.get('speaker_id')}).\n"
            f"{voice_status}\n"
            f"Updated transcript segments: {report.get('updated_segments')}.\n"
            f"{remaining_status}\n"
            "JSON/TXT/DOCX regenerated without retranscription."
        ),
        outputs.get("json"),
        outputs.get("txt"),
        outputs.get("docx"),
        unknown_dropdown,
        preview,
        "",
    )


def _transcript_speaker_info(source: Path) -> tuple[list[str], str]:
    data = _load_canonical(source)
    segments = data.get("segments") or []
    speakers = sorted(
        {
            str(segment.get("speaker"))
            for segment in segments
            if segment.get("speaker")
            and segment.get("speaker") != "UNKNOWN"
        }
    )
    lines: list[str] = []
    for speaker in speakers:
        example = _representative_segment(segments, speaker)
        text = str(example.get("text") or "").replace("\n", " ").strip()
        if len(text) > 150:
            text = text[:147] + "..."
        lines.append(
            f"{speaker}: {_format_time(example.get('start'))} — "
            f"{text or '[no text]'}"
        )
    return speakers, "\n".join(lines)


def load_registry_speakers(selected: str | None):
    source = _resolve_source(selected)
    speakers, preview = _transcript_speaker_info(source)
    return (
        gr.Dropdown(
            choices=speakers,
            value=speakers[0] if speakers else None,
            label="Confirmed speaker in selected recording",
        ),
        preview,
    )


def enroll_registry_speaker(
    selected: str | None,
    diarization_speaker: str | None,
    display_name: str | None,
) -> str:
    source = _resolve_source(selected)
    if not diarization_speaker:
        raise gr.Error("Load speakers and choose the speaker to save.")
    name = (display_name or "").strip()
    if not name:
        raise gr.Error("Enter a name for this speaker.")

    report = _run_json_tool(
        [
            APP_PYTHON,
            "-u",
            str(REGISTRY_TOOL_PATH),
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
    return (
        f"{action} identity: {report.get('display_name')} "
        f"({report.get('speaker_id')})\n"
        f"Source speaker: {report.get('diarization_speaker')}\n"
        f"Reference chunks added: {report.get('chunks_added')}\n"
        f"Stored reference embeddings: {report.get('reference_embeddings')}\n"
        f"Registry: {report.get('registry')}"
    )


def _recognition_summary(report: dict, *, applied: bool) -> str:
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
        if item.get("status") == "KNOWN":
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
    if applied:
        lines.extend(
            [
                "",
                f"Applied resolved names. Updated segments: {report.get('updated_segments', 0)}.",
                "JSON/TXT/DOCX regenerated beside the source recording.",
            ]
        )
    return "\n".join(lines)


def recognize_registry_speakers(selected: str | None) -> str:
    source = _resolve_source(selected)
    if not REGISTRY_PATH.is_file():
        raise gr.Error(
            "Speaker registry is empty. Save one confirmed speaker first."
        )
    report = _run_json_tool(
        [
            APP_PYTHON,
            "-u",
            str(REGISTRY_TOOL_PATH),
            "recognize",
            str(source),
            "--registry",
            str(REGISTRY_PATH),
        ],
        "Speaker recognition failed",
    )
    return _recognition_summary(report, applied=False)


def apply_registry_names(selected: str | None):
    source = _resolve_source(selected)
    if not REGISTRY_PATH.is_file():
        raise gr.Error(
            "Speaker registry is empty. Save one confirmed speaker first."
        )
    report = _run_json_tool(
        [
            APP_PYTHON,
            "-u",
            str(REGISTRY_TOOL_PATH),
            "apply",
            str(source),
            "--registry",
            str(REGISTRY_PATH),
        ],
        "Could not apply speaker names",
    )
    outputs = report.get("outputs") or {}
    return (
        _recognition_summary(report, applied=True),
        outputs.get("json"),
        outputs.get("txt"),
        outputs.get("docx"),
    )


def registry_overview() -> str:
    if not REGISTRY_PATH.is_file():
        return (
            "Registry is empty. It will be created at:\n"
            f"{REGISTRY_PATH}"
        )
    report = _run_json_tool(
        [
            APP_PYTHON,
            "-u",
            str(REGISTRY_TOOL_PATH),
            "list",
            "--registry",
            str(REGISTRY_PATH),
        ],
        "Could not read speaker registry",
    )
    lines = [
        f"Registry: {report.get('registry')}",
        f"Known identities: {report.get('known_identity_count')}",
    ]
    for identity in report.get("speakers") or []:
        lines.append(
            f"  {identity.get('display_name')} "
            f"({identity.get('speaker_id')}): "
            f"{identity.get('reference_embeddings')} reference embeddings, "
            f"{identity.get('sources')} source(s)"
        )
    return "\n".join(lines)


def _benchmark_summary(report: dict) -> str:
    same = report.get("same_speaker_similarity") or {}
    different = report.get("different_speaker_similarity") or {}
    holdout = report.get("closed_set_holdout") or {}
    calibration = report.get("calibration") or {}
    accuracy = holdout.get("accuracy")
    accuracy_text = (
        "n/a" if accuracy is None else f"{float(accuracy) * 100:.1f}%"
    )
    return (
        f"Embedding model: {report.get('embedding_model')}\n"
        f"Chunks per speaker: {report.get('chunks_per_speaker') or {}}\n"
        f"Same-speaker similarity: p10={_fmt_score(same.get('p10'))}, "
        f"median={_fmt_score(same.get('median'))}\n"
        f"Different-speaker similarity: p95={_fmt_score(different.get('p95'))}, "
        f"max={_fmt_score(different.get('max'))}\n"
        f"Conservative separation gap: "
        f"{_fmt_score(report.get('separation_gap_p10_vs_p95'))}\n"
        f"Closed-set holdout accuracy: {accuracy_text} "
        f"({holdout.get('correct', 0)}/{holdout.get('queries', 0)})\n"
        f"Exploratory threshold: "
        f"{_fmt_score(calibration.get('candidate_threshold'))}"
    )


def benchmark_speakers(selected: str | None) -> str:
    source = _resolve_source(selected)
    _load_canonical(source)
    report = _run_json_tool(
        [
            APP_PYTHON,
            "-u",
            str(BENCHMARK_PATH),
            str(source),
        ],
        "Speaker benchmark failed",
    )
    return _benchmark_summary(report)


def _cross_benchmark_summary(report: dict) -> str:
    reference = report.get("reference") or {}
    candidate = report.get("candidate") or {}
    matrix = report.get("similarity_matrix") or {}
    reference_speakers = sorted(
        (reference.get("chunks_per_speaker") or {}).keys()
    )
    lines = [
        f"Embedding model: {report.get('embedding_model')}",
        f"Reference: {Path(str(reference.get('source', ''))).name}",
        f"Second recording: {Path(str(candidate.get('source', ''))).name}",
        "",
        "Cross-recording cosine similarity matrix:",
    ]
    if reference_speakers and matrix:
        header = "candidate \\ reference | " + " | ".join(reference_speakers)
        lines.append(header)
        lines.append("-" * len(header))
        for candidate_speaker in sorted(matrix):
            row = matrix[candidate_speaker]
            scores = " | ".join(
                _fmt_score(row.get(ref)) for ref in reference_speakers
            )
            lines.append(f"{candidate_speaker} | {scores}")
    lines.extend(["", "Best matches (validation only):"])
    for match in report.get("best_matches") or []:
        lines.append(
            f"  {match.get('candidate_speaker')} -> "
            f"{match.get('reference_speaker')}: "
            f"score={_fmt_score(match.get('score'))}, "
            f"margin={_fmt_score(match.get('margin'))}, "
            f"mutual_best={'yes' if match.get('mutual_best') else 'no'}"
        )
    return "\n".join(lines)


def compare_recordings(
    reference_selected: str | None,
    candidate_selected: str | None,
) -> str:
    reference = _resolve_source(reference_selected)
    candidate = _resolve_source(candidate_selected)
    if reference == candidate:
        raise gr.Error("Choose two different recordings.")
    _load_canonical(reference)
    _load_canonical(candidate)
    report = _run_json_tool(
        [
            APP_PYTHON,
            "-u",
            str(CROSS_BENCHMARK_PATH),
            str(reference),
            str(candidate),
        ],
        "Cross-recording benchmark failed",
    )
    return _cross_benchmark_summary(report)


def build_ui() -> gr.Blocks:
    if not DRIVE_ROOT.exists():
        raise RuntimeError(f"Google Drive is not mounted: {DRIVE_ROOT}")

    runtime_mode = "GPU transcription mode" if GPU_AVAILABLE else "CPU maintenance mode"
    with gr.Blocks(title="Interview Transcriber") as demo:
        gr.Markdown(
            "# Interview Transcriber\n"
            "Choose an existing recording from Google Drive. "
            "Known voices are resolved automatically; only unknown speakers need names.\n\n"
            f"**Runtime:** {runtime_mode} · **Build:** `{REPO_HEAD}`"
        )
        recording = gr.FileExplorer(
            root_dir=str(DRIVE_ROOT),
            glob="**/*",
            file_count="single",
            label="Recording on Google Drive",
            height=420,
        )
        start = gr.Button(
            "Transcribe" if GPU_AVAILABLE else "Transcribe — GPU unavailable",
            variant="primary",
            interactive=GPU_AVAILABLE,
        )
        status = gr.Textbox(label="Status", interactive=False, lines=5)
        with gr.Row():
            json_output = gr.File(label="JSON")
            txt_output = gr.File(label="TXT")
            docx_output = gr.File(label="DOCX")

        with gr.Accordion("Unknown speakers — name once", open=True):
            gr.Markdown(
                "After transcription this section contains only unresolved speaker IDs. "
                "In CPU maintenance mode, use the load button to read them from an existing transcript. "
                "Choose one, verify the snippet, enter the name, and save it."
            )
            load_unknowns = gr.Button(
                "Load unknown speakers from existing transcript"
            )
            unknown_speaker = gr.Dropdown(
                choices=[],
                label="Unknown speaker",
            )
            unknown_preview = gr.Textbox(
                label="Representative snippet",
                interactive=False,
                lines=4,
            )
            unknown_name = gr.Textbox(
                label="Name",
                placeholder="Enter the speaker's name",
            )
            confirm_unknown = gr.Button(
                "Save name and update transcript",
                variant="primary",
            )
            unknown_status = gr.Textbox(
                label="Unknown-speaker status",
                interactive=False,
                lines=7,
            )

        with gr.Accordion("Speaker registry (advanced)", open=False):
            refresh_speakers = gr.Button(
                "Load all speakers from selected recording"
            )
            registry_speaker = gr.Dropdown(
                choices=[],
                label="Confirmed speaker in selected recording",
            )
            speaker_preview = gr.Textbox(
                label="Speaker examples",
                interactive=False,
                lines=6,
            )
            display_name = gr.Textbox(
                label="Name for confirmed speaker",
                placeholder="e.g. Alex",
            )
            save_speaker = gr.Button("Save confirmed speaker to registry")
            registry_status = gr.Textbox(
                label="Registry status",
                interactive=False,
                lines=7,
            )
            with gr.Row():
                recognize = gr.Button("Preview known-speaker matches")
                apply_names = gr.Button(
                    "Apply recognized names + regenerate files"
                )
                show_registry = gr.Button("Show registry")
            recognition_status = gr.Textbox(
                label="Speaker recognition",
                interactive=False,
                lines=16,
            )

        with gr.Accordion(
            "Speaker recognition benchmark (development)",
            open=False,
        ):
            benchmark = gr.Button("Benchmark selected recording")
            benchmark_status = gr.Textbox(
                label="Single-recording benchmark",
                interactive=False,
                lines=10,
            )
            comparison_recording = gr.FileExplorer(
                root_dir=str(DRIVE_ROOT),
                glob="**/*",
                file_count="single",
                label="Second transcribed recording",
                height=320,
            )
            compare = gr.Button("Compare speakers across recordings")
            compare_status = gr.Textbox(
                label="Cross-recording benchmark",
                interactive=False,
                lines=18,
            )

        start.click(
            fn=transcribe,
            inputs=recording,
            outputs=[
                status,
                json_output,
                txt_output,
                docx_output,
                unknown_speaker,
                unknown_preview,
                unknown_status,
            ],
            concurrency_limit=1,
        )
        load_unknowns.click(
            fn=load_existing_unknowns,
            inputs=recording,
            outputs=[unknown_speaker, unknown_preview, unknown_status],
        )
        unknown_speaker.change(
            fn=unknown_speaker_preview,
            inputs=[recording, unknown_speaker],
            outputs=unknown_preview,
        )
        confirm_unknown.click(
            fn=confirm_unknown_speaker,
            inputs=[recording, unknown_speaker, unknown_name],
            outputs=[
                unknown_status,
                json_output,
                txt_output,
                docx_output,
                unknown_speaker,
                unknown_preview,
                unknown_name,
            ],
            concurrency_limit=1,
        )

        refresh_speakers.click(
            fn=load_registry_speakers,
            inputs=recording,
            outputs=[registry_speaker, speaker_preview],
        )
        save_speaker.click(
            fn=enroll_registry_speaker,
            inputs=[recording, registry_speaker, display_name],
            outputs=registry_status,
            concurrency_limit=1,
        )
        recognize.click(
            fn=recognize_registry_speakers,
            inputs=recording,
            outputs=recognition_status,
            concurrency_limit=1,
        )
        apply_names.click(
            fn=apply_registry_names,
            inputs=recording,
            outputs=[
                recognition_status,
                json_output,
                txt_output,
                docx_output,
            ],
            concurrency_limit=1,
        )
        show_registry.click(
            fn=registry_overview,
            outputs=recognition_status,
        )
        benchmark.click(
            fn=benchmark_speakers,
            inputs=recording,
            outputs=benchmark_status,
            concurrency_limit=1,
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
    password = (
        os.environ.get("TRANSCRIBER_UI_PASSWORD")
        or secrets.token_urlsafe(12)
    )

    print("\nInterview Transcriber UI credentials", flush=True)
    print(f"Username: {username}", flush=True)
    print(f"Password: {password}", flush=True)
    print(
        "Keep the temporary Gradio URL and password private.\n",
        flush=True,
    )

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

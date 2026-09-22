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
CONFIRM_TOOL_PATH = Path(__file__).with_name("confirm_speaker.py").resolve()
RESOLUTION_TOOL_PATH = Path(__file__).with_name("speaker_resolution.py").resolve()
APP_PYTHON = os.environ.get("TRANSCRIBER_APP_PYTHON") or sys.executable
GPU_AVAILABLE = os.environ.get("TRANSCRIBER_GPU_AVAILABLE", "1") == "1"
REPO_HEAD = os.environ.get("TRANSCRIBER_REPO_HEAD", "unknown")
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
        raise gr.Error(
            f"Unsupported format {source.suffix}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return source


def _output_paths(source: Path) -> tuple[Path, Path, Path]:
    return tuple(
        source.with_name(f"{source.stem}_transcript.{ext}")
        for ext in ("json", "txt", "docx")
    )  # type: ignore[return-value]


def _format_time(value: object) -> str:
    if value is None:
        return "n/a"
    total = max(0, int(float(value)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _fmt_score(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


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
        last_lines = last_lines[-80:]
    if process.wait() != 0:
        raise gr.Error(
            f"{error_prefix}. Last log lines:\n\n" + "\n".join(last_lines[-20:])
        )


def _run_json_tool(command: list[str], error_prefix: str) -> dict:
    with tempfile.NamedTemporaryFile(
        prefix="transcriber-tool-", suffix=".json", delete=False
    ) as tmp:
        output = Path(tmp.name)
    try:
        _run_process([*command, "--output", str(output)], error_prefix)
        return json.loads(output.read_text(encoding="utf-8"))
    finally:
        output.unlink(missing_ok=True)


def _load_canonical(source: Path) -> dict:
    path = source.with_name(f"{source.stem}_transcript.json")
    if not path.is_file():
        raise gr.Error(
            "This recording has no canonical _transcript.json yet. "
            "Transcribe it when GPU is available."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _representative_segment(segments: list[dict], speaker: str) -> dict:
    candidates = [
        item
        for item in segments
        if str(item.get("speaker") or "UNKNOWN") == speaker
        and item.get("start") is not None
        and item.get("end") is not None
    ]
    return (
        max(
            candidates,
            key=lambda item: float(item["end"]) - float(item["start"]),
        )
        if candidates
        else {}
    )


def _unknown_speakers_from_data(data: dict) -> tuple[list[str], dict[str, str]]:
    segments = data.get("segments") or []
    concrete = sorted(
        {
            str(item.get("speaker"))
            for item in segments
            if item.get("speaker") and item.get("speaker") != "UNKNOWN"
        }
    )
    resolved = {
        str(item.get("speaker"))
        for item in segments
        if item.get("speaker")
        and item.get("speaker") != "UNKNOWN"
        and str(item.get("resolved_name") or "").strip()
    }
    resolution = {
        str(item.get("diarization_speaker")): item
        for item in (data.get("speaker_resolution") or {}).get("results") or []
    }
    unknowns = [speaker for speaker in concrete if speaker not in resolved]
    unknowns.sort(
        key=lambda speaker: (
            -int((resolution.get(speaker) or {}).get("chunks") or 0),
            speaker,
        )
    )

    previews: dict[str, str] = {}
    for speaker in unknowns:
        example = _representative_segment(segments, speaker)
        text = str(example.get("text") or "").replace("\n", " ").strip()
        if len(text) > 180:
            text = text[:177] + "..."
        item = resolution.get(speaker) or {}
        metadata: list[str] = []
        if item.get("chunks") is not None:
            metadata.append(f"usable chunks: {item.get('chunks')}")
        if item.get("matched_display_name"):
            metadata.append(
                f"best registry match: {item.get('matched_display_name')}"
            )
        if item.get("score") is not None:
            metadata.append(f"score: {_fmt_score(item.get('score'))}")
        if item.get("margin") is not None:
            metadata.append(f"margin: {_fmt_score(item.get('margin'))}")
        if item.get("reason"):
            metadata.append(f"recognition: {item.get('reason')}")
        suffix = "\n" + ", ".join(metadata) if metadata else ""
        previews[speaker] = (
            f"{speaker} at {_format_time(example.get('start'))}: "
            f"{text or '[no text]'}{suffix}"
        )
    return unknowns, previews


def _unknown_update(source: Path):
    unknowns, previews = _unknown_speakers_from_data(_load_canonical(source))
    selected = unknowns[0] if unknowns else None
    status = (
        f"Unknown speakers needing a name: {len(unknowns)} ({', '.join(unknowns)})."
        if unknowns
        else "No unresolved diarization speakers need a name."
    )
    return (
        gr.Dropdown(choices=unknowns, value=selected, label="Unknown speaker"),
        previews.get(selected, ""),
        status,
    )


def _registry_data() -> dict:
    return (
        json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        if REGISTRY_PATH.is_file()
        else {"speakers": []}
    )


def _usable_registry_identities() -> list[dict]:
    return [
        item
        for item in _registry_data().get("speakers") or []
        if item.get("embeddings")
    ]


def _registry_choices() -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    identities = sorted(
        _registry_data().get("speakers") or [],
        key=lambda item: (
            str(item.get("display_name") or "").casefold(),
            str(item.get("speaker_id") or ""),
        ),
    )
    for item in identities:
        speaker_id = str(item.get("speaker_id") or "").strip()
        name = str(item.get("display_name") or "").strip()
        if speaker_id and name:
            choices.append(
                (
                    f"{name} · {speaker_id} · "
                    f"{len(item.get('embeddings') or [])} voice refs",
                    speaker_id,
                )
            )
    return choices


def _identity_dropdown():
    return gr.Dropdown(
        choices=_registry_choices(),
        value=None,
        label="Existing person (optional)",
    )


def _registry_name(speaker_id: str | None) -> str:
    for item in _registry_data().get("speakers") or []:
        if str(item.get("speaker_id") or "") == str(speaker_id or ""):
            return str(item.get("display_name") or "")
    return ""


def _result_summary(data: dict) -> str:
    bits = ["Transcript ready."]
    if data.get("speaker_count") is not None:
        bits.append(f"Detected speakers: {data['speaker_count']}.")
    segments = data.get("segments") or []
    if segments:
        bits.append(f"Segments: {len(segments)}.")
    unknowns, _ = _unknown_speakers_from_data(data)
    bits.append(
        "Unknown speakers: " + ", ".join(unknowns) + "."
        if unknowns
        else "All diarization speakers are resolved."
    )
    return " ".join(bits)


def _start_result(source: Path, status: str):
    json_path, txt_path, docx_path = _output_paths(source)
    unknown_dropdown, preview, unknown_status = _unknown_update(source)
    return (
        status,
        str(json_path),
        str(txt_path) if txt_path.is_file() else None,
        str(docx_path) if docx_path.is_file() else None,
        unknown_dropdown,
        preview,
        unknown_status,
        _identity_dropdown(),
        "",
    )


def load_existing_transcript(selected: str | None):
    source = _resolve_source(selected)
    _, txt_path, docx_path = _output_paths(source)
    data = _load_canonical(source)
    missing = [
        str(path) for path in (txt_path, docx_path) if not path.is_file()
    ]
    if missing:
        raise gr.Error(
            "Expected transcript output files are missing: " + ", ".join(missing)
        )
    return _start_result(source, _result_summary(data))


def _recognition_summary(report: dict) -> str:
    thresholds = report.get("thresholds") or {}
    results = report.get("results") or []
    resolved = sum(
        1
        for item in results
        if item.get("status") in {"KNOWN", "CONFIRMED"}
    )
    lines = [
        "Known-speaker recognition refreshed from the current registry.",
        (
            "Thresholds: "
            f"similarity>={thresholds.get('similarity', 0.60)}, "
            f"margin>={thresholds.get('margin', 0.20)}, "
            f"min_chunks={thresholds.get('min_chunks', 2)}"
        ),
        f"Resolved diarization IDs: {resolved}/{len(results)}",
        "",
    ]
    for item in results:
        speaker = item.get("diarization_speaker")
        status = item.get("status")
        if status in {"KNOWN", "CONFIRMED"}:
            name = item.get("resolved_name") or item.get("matched_display_name")
            lines.append(
                f"{speaker} -> {name} [{status}] "
                f"score={_fmt_score(item.get('score'))}, "
                f"margin={_fmt_score(item.get('margin'))}, "
                f"chunks={item.get('chunks')}, reason={item.get('reason')}"
            )
        else:
            lines.append(
                f"{speaker} -> UNKNOWN "
                f"best={item.get('matched_display_name') or 'n/a'}, "
                f"score={_fmt_score(item.get('score'))}, "
                f"margin={_fmt_score(item.get('margin'))}, "
                f"chunks={item.get('chunks')}, reason={item.get('reason')}"
            )
    lines.extend(
        ["", "JSON/TXT/DOCX regenerated without WhisperX retranscription."]
    )
    return "\n".join(lines)


def rerun_known_speaker_recognition(
    selected: str | None,
) -> Iterator[tuple]:
    source = _resolve_source(selected)
    _load_canonical(source)
    if not REGISTRY_PATH.is_file():
        raise gr.Error(
            "Speaker registry is empty. Confirm at least one speaker first."
        )
    identities = _usable_registry_identities()
    if not identities:
        raise gr.Error(
            "The registry has no identities with usable voice references yet."
        )

    yield _start_result(
        source,
        (
            f"Re-recognizing speakers against {len(identities)} registry "
            "identities... WhisperX/diarization will not run."
        ),
    )
    report = _run_json_tool(
        [
            APP_PYTHON,
            "-u",
            str(RESOLUTION_TOOL_PATH),
            str(source),
            "--registry",
            str(REGISTRY_PATH),
        ],
        "Known-speaker re-recognition failed",
    )
    outputs = report.get("outputs") or {}
    result = list(_start_result(source, _recognition_summary(report)))
    json_path, txt_path, docx_path = _output_paths(source)
    result[1] = outputs.get("json") or str(json_path)
    result[2] = outputs.get("txt") or str(txt_path)
    result[3] = outputs.get("docx") or str(docx_path)
    yield tuple(result)


def transcribe(selected: str | None) -> Iterator[tuple]:
    if not GPU_AVAILABLE:
        raise gr.Error(
            "GPU is unavailable. Use Load existing transcript or "
            "Re-recognize known speakers for existing files."
        )
    source = _resolve_source(selected)
    if not os.environ.get("HF_TOKEN"):
        raise gr.Error(
            "HF_TOKEN is not available. Restart the launcher and check Colab Secrets."
        )
    yield (
        "Processing...",
        None,
        None,
        None,
        gr.Dropdown(choices=[], value=None, label="Unknown speaker"),
        "",
        "Unknown speakers will appear here after transcription.",
        _identity_dropdown(),
        "",
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
    yield _start_result(source, _result_summary(_load_canonical(source)))


def unknown_speaker_preview(
    selected: str | None,
    diarization_speaker: str | None,
) -> str:
    source = _resolve_source(selected)
    if not diarization_speaker:
        return ""
    unknowns, previews = _unknown_speakers_from_data(_load_canonical(source))
    if diarization_speaker not in unknowns:
        return "This speaker is already resolved."
    return previews.get(diarization_speaker, "")


def existing_identity_selected(speaker_id: str | None) -> str:
    if not speaker_id:
        return ""
    return (
        "Will link this diarization speaker to existing identity: "
        f"{_registry_name(speaker_id)} ({speaker_id}). "
        "The New person name field is ignored while an existing person is selected."
    )


def confirm_unknown_speaker(
    selected: str | None,
    diarization_speaker: str | None,
    existing_speaker_id: str | None,
    display_name: str | None,
):
    source = _resolve_source(selected)
    if not diarization_speaker:
        raise gr.Error("Choose an unknown speaker first.")
    existing_speaker_id = (existing_speaker_id or "").strip() or None
    name = (display_name or "").strip()
    if not existing_speaker_id and not name:
        raise gr.Error(
            "Choose an existing person, or enter a name for a new person."
        )
    unknowns, _ = _unknown_speakers_from_data(_load_canonical(source))
    if diarization_speaker not in unknowns:
        raise gr.Error(
            f"{diarization_speaker} is no longer unresolved. Reload the transcript."
        )

    command = [
        APP_PYTHON,
        "-u",
        str(CONFIRM_TOOL_PATH),
        str(source),
        diarization_speaker,
        "--registry",
        str(REGISTRY_PATH),
    ]
    command.extend(
        ["--existing-speaker-id", existing_speaker_id]
        if existing_speaker_id
        else ["--display-name", name]
    )
    report = _run_json_tool(command, "Could not confirm unknown speaker")
    outputs = report.get("outputs") or {}
    unknown_dropdown, preview, remaining_status = _unknown_update(source)
    if report.get("voice_reference_saved", False):
        voice_status = (
            f"Voice reference saved: {report.get('chunks_added')} usable chunk(s); "
            f"identity now has {report.get('reference_embeddings')} stored "
            "reference embedding(s)."
        )
    else:
        voice_status = (
            "No voice reference was added because this diarization speaker "
            "has too little usable speech."
        )
    action = (
        "Linked to existing identity"
        if report.get("linked_existing")
        else "Created new identity"
    )
    return (
        f"{action}: {report.get('display_name')} ({report.get('speaker_id')}).\n"
        f"{voice_status}\n"
        f"Updated transcript segments: {report.get('updated_segments')}.\n"
        f"{remaining_status}\n"
        "JSON/TXT/DOCX regenerated without retranscription.",
        outputs.get("json"),
        outputs.get("txt"),
        outputs.get("docx"),
        unknown_dropdown,
        preview,
        _identity_dropdown(),
        "",
        "",
    )


def registry_overview() -> str:
    identities = _registry_data().get("speakers") or []
    lines = [
        f"Registry: {REGISTRY_PATH}",
        f"Known identities: {len(identities)}",
    ]
    for item in sorted(
        identities,
        key=lambda value: str(value.get("display_name") or "").casefold(),
    ):
        lines.append(
            f"{item.get('display_name')} ({item.get('speaker_id')}): "
            f"{len(item.get('embeddings') or [])} voice refs, "
            f"{len(item.get('provenance') or [])} provenance record(s)"
        )
    return "\n".join(lines)


def refresh_existing_people():
    return _identity_dropdown()


def build_ui() -> gr.Blocks:
    if not DRIVE_ROOT.exists():
        raise RuntimeError(f"Google Drive is not mounted: {DRIVE_ROOT}")
    runtime_mode = (
        "GPU transcription mode" if GPU_AVAILABLE else "CPU maintenance mode"
    )
    with gr.Blocks(title="Interview Transcriber") as demo:
        gr.Markdown(
            "# Interview Transcriber\n"
            "Choose a recording from Google Drive. Known voices are resolved "
            "automatically; unknown speakers can be linked to an existing person "
            "or saved as a new person.\n\n"
            f"**Runtime:** {runtime_mode} · **Build:** `{REPO_HEAD}`"
        )
        recording = gr.FileExplorer(
            root_dir=str(DRIVE_ROOT),
            glob="**/*",
            file_count="single",
            label="Recording on Google Drive",
            height=420,
        )
        with gr.Row():
            start = gr.Button(
                "Transcribe" if GPU_AVAILABLE else "Transcribe — GPU unavailable",
                variant="primary",
                interactive=GPU_AVAILABLE,
            )
            load_existing = gr.Button("Load existing transcript")
            rerecognize = gr.Button("Re-recognize known speakers")
        gr.Markdown(
            "**Re-recognize known speakers** uses the current registry and existing "
            "transcript. It recalculates voice matching and regenerates JSON/TXT/DOCX "
            "without running WhisperX or diarization again."
        )
        status = gr.Textbox(label="Status", interactive=False, lines=10)
        with gr.Row():
            json_output = gr.File(label="JSON")
            txt_output = gr.File(label="TXT")
            docx_output = gr.File(label="DOCX")

        with gr.Accordion("Unknown speakers — name once", open=True):
            gr.Markdown(
                "Choose an unresolved diarization ID. Link it to an existing person "
                "when appropriate; otherwise create a new identity."
            )
            unknown_speaker = gr.Dropdown(
                choices=[], label="Unknown speaker"
            )
            unknown_preview = gr.Textbox(
                label="Representative snippet + recognition diagnostics",
                interactive=False,
                lines=6,
            )
            with gr.Row():
                existing_identity = gr.Dropdown(
                    choices=_registry_choices(),
                    value=None,
                    label="Existing person (optional)",
                )
                refresh_people = gr.Button("Refresh known people")
            identity_hint = gr.Textbox(
                label="Identity action",
                interactive=False,
                lines=2,
            )
            unknown_name = gr.Textbox(
                label="New person name",
                placeholder="Required only when Existing person is blank",
            )
            confirm_unknown = gr.Button(
                "Link / create identity and update transcript",
                variant="primary",
            )
            unknown_status = gr.Textbox(
                label="Unknown-speaker status",
                interactive=False,
                lines=7,
            )

        with gr.Accordion("Speaker registry", open=False):
            show_registry = gr.Button("Show registry")
            registry_status = gr.Textbox(
                label="Registry", interactive=False, lines=12
            )

        start_outputs = [
            status,
            json_output,
            txt_output,
            docx_output,
            unknown_speaker,
            unknown_preview,
            unknown_status,
            existing_identity,
            identity_hint,
        ]
        start.click(
            fn=transcribe,
            inputs=recording,
            outputs=start_outputs,
            concurrency_limit=1,
        )
        load_existing.click(
            fn=load_existing_transcript,
            inputs=recording,
            outputs=start_outputs,
        )
        rerecognize.click(
            fn=rerun_known_speaker_recognition,
            inputs=recording,
            outputs=start_outputs,
            concurrency_limit=1,
        )
        unknown_speaker.change(
            fn=unknown_speaker_preview,
            inputs=[recording, unknown_speaker],
            outputs=unknown_preview,
        )
        existing_identity.change(
            fn=existing_identity_selected,
            inputs=existing_identity,
            outputs=identity_hint,
        )
        refresh_people.click(
            fn=refresh_existing_people,
            outputs=existing_identity,
        )
        confirm_unknown.click(
            fn=confirm_unknown_speaker,
            inputs=[
                recording,
                unknown_speaker,
                existing_identity,
                unknown_name,
            ],
            outputs=[
                unknown_status,
                json_output,
                txt_output,
                docx_output,
                unknown_speaker,
                unknown_preview,
                existing_identity,
                unknown_name,
                identity_hint,
            ],
            concurrency_limit=1,
        )
        show_registry.click(
            fn=registry_overview,
            outputs=registry_status,
        )
    return demo


def main() -> None:
    username = "transcriber"
    password = (
        os.environ.get("TRANSCRIBER_UI_PASSWORD") or secrets.token_urlsafe(12)
    )
    print("\nInterview Transcriber UI credentials", flush=True)
    print(f"Username: {username}", flush=True)
    print(f"Password: {password}", flush=True)
    print("The UI will open inside this Colab notebook. Keep the password private.\n", flush=True)
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        auth=(username, password),
        allowed_paths=[str(DRIVE_ROOT)],
        show_error=True,
    )


if __name__ == "__main__":
    main()

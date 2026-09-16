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
        _run_process([*command, "--output", str(report_path)], error_prefix)
        return json.loads(report_path.read_text(encoding="utf-8"))
    finally:
        report_path.unlink(missing_ok=True)


def _load_canonical(source: Path) -> dict:
    json_path = source.with_name(f"{source.stem}_transcript.json")
    if not json_path.is_file():
        raise gr.Error(
            "This recording has no canonical _transcript.json yet. "
            "Transcribe it when GPU is available."
        )
    return json.loads(json_path.read_text(encoding="utf-8"))


def _representative_segment(segments: list[dict], speaker: str) -> dict:
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


def _unknown_speakers_from_data(data: dict) -> tuple[list[str], dict[str, str]]:
    segments = data.get("segments") or []
    concrete = sorted(
        {
            str(segment.get("speaker"))
            for segment in segments
            if segment.get("speaker") and segment.get("speaker") != "UNKNOWN"
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
    status = (
        f"Unknown speakers needing a name: {len(unknowns)} ({', '.join(unknowns)})."
        if unknowns
        else "No unresolved diarization speakers need a name."
    )
    return (
        gr.Dropdown(choices=unknowns, value=selected, label="Unknown speaker"),
        preview,
        status,
    )


def _registry_data() -> dict:
    if not REGISTRY_PATH.is_file():
        return {"speakers": []}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _registry_choices() -> list[tuple[str, str]]:
    identities = _registry_data().get("speakers") or []
    choices: list[tuple[str, str]] = []
    for identity in sorted(
        identities,
        key=lambda item: (
            str(item.get("display_name") or "").casefold(),
            str(item.get("speaker_id") or ""),
        ),
    ):
        speaker_id = str(identity.get("speaker_id") or "").strip()
        name = str(identity.get("display_name") or "").strip()
        if speaker_id and name:
            refs = len(identity.get("embeddings") or [])
            choices.append((f"{name} · {speaker_id} · {refs} voice refs", speaker_id))
    return choices


def _registry_name(speaker_id: str | None) -> str:
    if not speaker_id:
        return ""
    for identity in _registry_data().get("speakers") or []:
        if str(identity.get("speaker_id") or "") == speaker_id:
            return str(identity.get("display_name") or "")
    return ""


def load_existing_transcript(selected: str | None):
    source = _resolve_source(selected)
    json_path, txt_path, docx_path = _output_paths(source)
    data = _load_canonical(source)
    unknown_dropdown, preview, unknown_status = _unknown_update(source)
    missing = [str(path) for path in (txt_path, docx_path) if not path.is_file()]
    if missing:
        raise gr.Error("Expected transcript output files are missing: " + ", ".join(missing))
    return (
        _result_summary(data),
        str(json_path),
        str(txt_path),
        str(docx_path),
        unknown_dropdown,
        preview,
        unknown_status,
        gr.Dropdown(
            choices=_registry_choices(),
            value=None,
            label="Existing person (optional)",
        ),
        "",
    )


def _result_summary(data: dict) -> str:
    bits = ["Transcript ready."]
    if data.get("speaker_count") is not None:
        bits.append(f"Detected speakers: {data['speaker_count']}.")
    segments = data.get("segments") or []
    if segments:
        bits.append(f"Segments: {len(segments)}.")
    unknowns, _ = _unknown_speakers_from_data(data)
    if unknowns:
        bits.append("Unknown speakers: " + ", ".join(unknowns) + ".")
    else:
        bits.append("All diarization speakers are resolved.")
    return " ".join(bits)


def transcribe(selected: str | None) -> Iterator[tuple]:
    if not GPU_AVAILABLE:
        raise gr.Error(
            "GPU is unavailable in this Colab runtime. Use 'Load existing transcript' "
            "for maintenance, or start a GPU runtime later to transcribe a new recording."
        )
    source = _resolve_source(selected)
    if not os.environ.get("HF_TOKEN"):
        raise gr.Error("HF_TOKEN is not available. Restart the launcher and check Colab Secrets.")

    yield (
        "Processing...",
        None,
        None,
        None,
        gr.Dropdown(choices=[], value=None, label="Unknown speaker"),
        "",
        "Unknown speakers will appear here after transcription.",
        gr.Dropdown(
            choices=_registry_choices(),
            value=None,
            label="Existing person (optional)",
        ),
        "",
    )
    _run_process([APP_PYTHON, "-u", str(APP_PATH), str(source)], "Transcription failed")

    json_path, txt_path, docx_path = _output_paths(source)
    missing = [str(path) for path in (json_path, txt_path, docx_path) if not path.exists()]
    if missing:
        raise gr.Error(
            "Pipeline finished but expected output files are missing: " + ", ".join(missing)
        )

    data = _load_canonical(source)
    unknown_dropdown, preview, unknown_status = _unknown_update(source)
    yield (
        _result_summary(data),
        str(json_path),
        str(txt_path),
        str(docx_path),
        unknown_dropdown,
        preview,
        unknown_status,
        gr.Dropdown(
            choices=_registry_choices(),
            value=None,
            label="Existing person (optional)",
        ),
        "",
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


def existing_identity_selected(speaker_id: str | None) -> str:
    if not speaker_id:
        return ""
    name = _registry_name(speaker_id)
    return (
        f"Will link this diarization speaker to existing identity: {name} ({speaker_id}). "
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
        raise gr.Error("Choose an existing person, or enter a name for a new person.")

    data = _load_canonical(source)
    unknowns, _ = _unknown_speakers_from_data(data)
    if diarization_speaker not in unknowns:
        raise gr.Error(
            f"{diarization_speaker} is no longer unresolved. Reload the transcript and choose another speaker."
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
    if existing_speaker_id:
        command.extend(["--existing-speaker-id", existing_speaker_id])
    else:
        command.extend(["--display-name", name])

    report = _run_json_tool(command, "Could not confirm unknown speaker")
    outputs = report.get("outputs") or {}
    unknown_dropdown, preview, remaining_status = _unknown_update(source)

    if report.get("voice_reference_saved", False):
        voice_status = (
            f"Voice reference saved: {report.get('chunks_added')} usable chunk(s); "
            f"identity now has {report.get('reference_embeddings')} stored reference embedding(s)."
        )
    else:
        voice_status = (
            "No voice reference was added because this diarization speaker has too little usable speech. "
            "The confirmed identity/name is still stored and the transcript is updated."
        )

    identity_action = (
        "Linked to existing identity"
        if report.get("linked_existing")
        else "Created new identity"
    )
    return (
        (
            f"{identity_action}: {report.get('display_name')} ({report.get('speaker_id')}).\n"
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
        gr.Dropdown(
            choices=_registry_choices(),
            value=None,
            label="Existing person (optional)",
        ),
        "",
        "",
    )


def registry_overview() -> str:
    identities = _registry_data().get("speakers") or []
    lines = [f"Registry: {REGISTRY_PATH}", f"Known identities: {len(identities)}"]
    for identity in sorted(
        identities,
        key=lambda item: str(item.get("display_name") or "").casefold(),
    ):
        lines.append(
            f"{identity.get('display_name')} ({identity.get('speaker_id')}): "
            f"{len(identity.get('embeddings') or [])} voice refs, "
            f"{len(identity.get('provenance') or [])} provenance record(s)"
        )
    return "\n".join(lines)


def refresh_existing_people():
    return gr.Dropdown(
        choices=_registry_choices(),
        value=None,
        label="Existing person (optional)",
    )


def build_ui() -> gr.Blocks:
    if not DRIVE_ROOT.exists():
        raise RuntimeError(f"Google Drive is not mounted: {DRIVE_ROOT}")

    runtime_mode = "GPU transcription mode" if GPU_AVAILABLE else "CPU maintenance mode"
    with gr.Blocks(title="Interview Transcriber") as demo:
        gr.Markdown(
            "# Interview Transcriber\n"
            "Choose a recording from Google Drive. Known voices are resolved automatically; "
            "unknown diarization speakers can be linked to an existing person or saved as a new person.\n\n"
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

        status = gr.Textbox(label="Status", interactive=False, lines=4)
        with gr.Row():
            json_output = gr.File(label="JSON")
            txt_output = gr.File(label="TXT")
            docx_output = gr.File(label="DOCX")

        with gr.Accordion("Unknown speakers — name once", open=True):
            gr.Markdown(
                "For each unresolved diarization ID, first decide whether it is someone already in the registry. "
                "If yes, choose that person below. If not, leave Existing person blank and enter a new name."
            )
            unknown_speaker = gr.Dropdown(choices=[], label="Unknown speaker")
            unknown_preview = gr.Textbox(
                label="Representative snippet",
                interactive=False,
                lines=4,
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
                label="Registry",
                interactive=False,
                lines=12,
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
        show_registry.click(fn=registry_overview, outputs=registry_status)

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

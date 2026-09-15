from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docx import Document


def _stamp(seconds: float | int | None) -> str:
    value = max(0, int(seconds or 0))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _normalized_segment(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "start": segment.get("start"),
        "end": segment.get("end"),
        "speaker": segment.get("speaker", "UNKNOWN"),
        "resolved_name": None,
        "text": (segment.get("text") or "").strip(),
        "words": segment.get("words", []),
    }


def export_transcript(result: dict[str, Any], source: Path) -> dict[str, Path]:
    base = source.with_suffix("")
    txt_path = Path(f"{base}_transcript.txt")
    docx_path = Path(f"{base}_transcript.docx")
    json_path = Path(f"{base}_transcript.json")

    segments = [_normalized_segment(segment) for segment in result.get("segments", [])]

    canonical = {
        "source": {
            "filename": source.name,
            "path": str(source),
        },
        "processed_at": result.get("processed_at"),
        "language": result.get("language"),
        "models": result.get("models", {}),
        "speaker_count": result.get("speaker_count", 0),
        "detected_speakers": result.get("detected_speakers", []),
        "segments": segments,
    }
    json_path.write_text(
        json.dumps(canonical, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = []
    for segment in segments:
        lines.append(
            f"[{_stamp(segment['start'])}–{_stamp(segment['end'])}] "
            f"{segment['speaker']}: {segment['text']}"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    document = Document()
    document.add_heading(source.stem, level=1)
    for segment in segments:
        paragraph = document.add_paragraph()
        paragraph.add_run(
            f"[{_stamp(segment['start'])}–{_stamp(segment['end'])}] "
            f"{segment['speaker']}: "
        ).bold = True
        paragraph.add_run(segment["text"])
    document.save(docx_path)

    return {"json": json_path, "txt": txt_path, "docx": docx_path}

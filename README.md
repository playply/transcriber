# Interview Transcriber MVP

Super-lean clean-room transcription baseline for Russian interview recordings.

Current scope is intentionally limited to one vertical slice:

1. run in Google Colab with a T4 GPU;
2. read an existing recording from mounted Google Drive;
3. transcribe in Russian with WhisperX;
4. align timestamps;
5. diarize speakers with pyannote and automatically determine speaker count;
6. save `*_transcript.docx`, `*_transcript.txt`, and canonical `*_transcript.json` beside the source recording.

Gradio UI and persistent speaker recognition are deliberately not implemented yet.

## Security

`HF_TOKEN` must be a Hugging Face **Read** token stored in Colab Secrets under the name `HF_TOKEN`. The token is read at runtime only. Never commit it, put it in Drive, hard-code it in the notebook, or print it to logs.

## Files

- `app.py` — command-line entry point.
- `pipeline.py` — WhisperX transcription, alignment, and pyannote diarization.
- `exporters.py` — TXT, DOCX, and canonical JSON exporters.
- `speaker_registry.py` — explicit placeholder for the later speaker-recognition milestone.
- `requirements.txt` — tested baseline dependency pins.
- `colab_launcher.ipynb` — fresh-runtime launcher for Colab.

## Colab clean-room run

1. Open `colab_launcher.ipynb` in Google Colab.
2. Select a T4 GPU runtime.
3. Add `HF_TOKEN` to Colab Secrets and allow notebook access to it.
4. Run the cells from top to bottom.
5. The launcher clones or refreshes `main`, installs dependencies, verifies CUDA, mounts Drive, and loads the token without printing it.
6. Change only `SOURCE_PATH` to an existing `.mp4`, `.mp3`, `.m4a`, or `.wav` recording in Drive.
7. Run the transcription cell.
8. Run the final verification cell; it fails if any DOCX/TXT/JSON output is missing.

Example source:

```text
/content/drive/MyDrive/Interviews/meeting.m4a
```

Expected outputs in the same directory:

```text
meeting_transcript.docx
meeting_transcript.txt
meeting_transcript.json
```

If imports fail immediately after package installation with a NumPy/import compatibility error, restart the Colab runtime once and rerun the notebook from the first cell. This recovery was needed in a previously observed Colab runtime transition.

## What the baseline records

The canonical JSON preserves:

- source filename and path;
- processing timestamp;
- transcription language;
- Whisper model name;
- diarization model name;
- installed versions of WhisperX, torch, pyannote.audio, and transformers;
- detected diarization speaker IDs;
- speaker count excluding `UNKNOWN`;
- segment start/end timestamps;
- original diarization speaker ID;
- `resolved_name` placeholder;
- text and word timestamps when available.

## T4 memory behavior

The pipeline releases the Whisper transcription model before alignment and releases the alignment model before diarization. This keeps the clean-room baseline leaner on a Colab T4 without changing the transcription stack.

## Local CLI shape

The application entry point is intentionally simple:

```bash
python app.py /path/to/meeting.m4a
```

`HF_TOKEN` must already exist in the process environment. The baseline refuses to run transcription when CUDA is unavailable because CPU execution is outside the MVP runtime target.

## Baseline notes

The initial pins reflect the previously tested Colab setup (`whisperx==3.8.5`, `transformers==4.57.6`, `python-docx==1.2.0`). They remain a clean-room starting point, not a permanent compatibility contract. Re-pin only after a fresh Colab run is verified end-to-end.

The repository can prepare and validate the launcher, but the actual clean-room acceptance run still has to be executed in a fresh Colab T4 runtime against a real Drive recording.

## Not in this milestone

- Gradio or another web UI
- speaker registry persistence
- voice embeddings / known-speaker matching
- Telegram
- Drive folder watching
- permanent hosting
- batch processing or dashboards

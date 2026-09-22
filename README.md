# Interview Transcriber MVP

Super-lean clean-room transcription baseline for Russian interview recordings.

Current MVP scope is intentionally limited to one recording per user session:

1. start a temporary worker in Google Colab;
2. choose one existing recording from mounted Google Drive;
3. use one primary **Process / Continue** action that detects whether the file needs transcription or already has a canonical transcript;
4. transcribe in Russian with WhisperX when needed;
5. align timestamps and diarize speakers with pyannote;
6. recognize high-confidence known speakers from the persistent Drive registry;
7. ask the user to resolve only meaningful unknown speakers;
8. save `*_transcript.docx`, `*_transcript.txt`, and canonical `*_transcript.json` beside the source recording.

Multi-file processing, persistent job queues, and automatic Drive folder watching are intentionally out of scope. The automation target is the single-file workflow around startup, state detection, speaker resolution, finalization, and GPU release.

## Security

`HF_TOKEN` must be a Hugging Face **Read** token stored in Colab Secrets under the name `HF_TOKEN`. The token is read at runtime only. Never commit it, put it in Drive, hard-code it in the notebook, or print it to logs.

## Files

- `app.py` — transcription entry point plus automatic registry matching after transcription.
- `pipeline.py` — WhisperX transcription, alignment, and pyannote diarization.
- `exporters.py` — TXT, DOCX, and canonical JSON exporters.
- `speaker_registry.py` — persistent speaker registry and embedding storage logic.
- `speaker_resolution.py` — known-speaker matching and transcript regeneration.
- `confirm_speaker.py` — explicit unknown-speaker confirmation/linking.
- `ui.py` — Gradio single-file workflow and maintenance tools.
- `requirements.txt` — tested baseline dependency pins.
- `colab_launcher.ipynb` / `colab_bootstrap.py` — one-cell Colab startup, Drive mount, secrets, dependencies, and automatic Gradio launch.

## Colab clean-room run

1. Open `colab_launcher.ipynb` in Google Colab.
2. Select a T4 GPU runtime for a new transcription. CPU mode is sufficient for maintenance on an existing transcript.
3. Add `HF_TOKEN` to Colab Secrets and allow notebook access to it.
4. Run the single **START INTERVIEW TRANSCRIBER** cell.
5. The launcher refreshes the repository, mounts Drive, loads the token without printing it, prepares dependencies, launches the temporary Gradio UI, and attempts to open it automatically in a new browser tab.
6. The UI uses a short-lived magic-link to establish a secure runtime session cookie, so there is no separate username/password form. If the browser blocks automatic tab opening, click the single **Open Interview Transcriber** button shown by the START cell.
7. In Gradio, choose one existing `.mp4`, `.mp3`, `.m4a`, or `.wav` recording from Drive.
8. Press **Process / Continue**. If canonical JSON already exists, the app opens it without WhisperX retranscription; otherwise it starts the GPU transcription pipeline.
9. Resolve any meaningful unknown speakers. JSON/TXT/DOCX are regenerated beside the source recording without retranscription.

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

## Explicitly out of scope

- multi-file transcription in one run
- persistent transcription queues
- automatic Drive folder watching
- Telegram
- permanent hosting
- native Mac/iOS applications
- paid transcription SaaS
- dashboards, accounts, or multi-user orchestration

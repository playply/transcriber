# Interview Transcriber — Project Status Snapshot

**Snapshot date:** 2026-09-22  
**Repository:** `playply/transcriber`  
**Application baseline before this snapshot:** `3f94be208df1d7341d7b9d8a5d2dc43c44d5898a`

## 1. MVP goal

Build the smallest reliable single-file workflow that takes one interview recording already stored on Google Drive, transcribes it in Russian when needed, diarizes speakers, recognizes previously known speakers from a persistent registry, asks the user to name only unresolved speakers that have enough usable speech to identify, and saves canonical JSON plus TXT and DOCX beside the source recording. One Colab session is intentionally optimized around one selected recording; multi-file processing and persistent transcription queues are out of scope.

Out of scope for the MVP: permanent hosting, Telegram integration, native Mac/iOS apps, paid transcription SaaS, and a large dashboard.

## 2. Current stack

- Runtime: Google Colab, T4 for new transcription and CPU maintenance mode for existing transcripts.
- Transcription/alignment: WhisperX `3.8.5`.
- Diarization: pyannote speaker diarization.
- Voice embeddings: `pyannote/wespeaker-voxceleb-resnet34-LM`.
- Transformers: `4.57.6`.
- DOCX generation: `python-docx 1.2.0`.
- UI: temporary authenticated Gradio `6.27.0` launched from Colab.
- Storage: recordings, transcripts, and speaker registry on Google Drive; code in GitHub.
- Hugging Face token: read-only and loaded from Colab Secrets; never stored in GitHub, Drive outputs, logs, or transcript files.

## 3. Output contract

For a source recording such as `meeting.mp4`, the system creates in the same Google Drive folder:

- `meeting_transcript.json`
- `meeting_transcript.txt`
- `meeting_transcript.docx`

JSON is the canonical record and preserves timestamps, original diarization speaker IDs, resolved display names, model metadata, and speaker-resolution audit information.

## 4. Current speaker-recognition policy

The current conservative automatic-match gates are:

- cosine similarity `>= 0.60`
- margin over runner-up `>= 0.20` when a runner-up exists
- at least `2` usable voice chunks

The matcher allows multiple diarization IDs in one recording to resolve to the same known identity when each independently passes the core gates.

Diarization IDs with insufficient usable speech are preserved in the transcript but are no longer shown to the user as identities that require naming.

## 5. Unseen-recording validation completed

A fresh unseen recording was processed using the ordinary `Transcribe` flow before any manual naming.

Observed result:

- 4 diarization IDs were produced.
- 2 previously registered identities were resolved automatically with real scores, margins, and 6 usable chunks each.
- Automatic match A: score approximately `0.804`, margin approximately `0.455`.
- Automatic match B: score approximately `0.751`, margin approximately `0.358`.
- Both automatic assignments were manually verified as correct.
- A third primary participant was selected as the best registry candidate but remained `UNKNOWN` because score approximately `0.491` and margin approximately `0.193` were below the conservative gates. This was a safe false negative rather than a false positive.
- That unresolved diarization ID was then explicitly linked to the correct existing registry identity in CPU maintenance mode. Six usable chunks were saved, bringing that identity to the registry reference cap.
- A fourth diarization ID had zero usable voice chunks and was correctly excluded from the naming UX after the UI hardening change.

This validates the central MVP behavior: known speakers can be recognized automatically across recordings, uncertain matches remain unknown, and the user only needs to confirm meaningful unresolved identities.

## 6. UI hardening completed

Commit `3f94be2` changed the unknown-speaker UI so diarization fragments with `insufficient_usable_speech` or fewer usable chunks than the configured minimum are not presented as speakers the user must name.

The underlying transcript segments and diarization IDs remain intact in canonical JSON; only the naming workflow is filtered.

CPU maintenance verification confirmed that the test recording now presents only the meaningful unresolved speaker for confirmation.

## 7. Colab access and GPU conservation

The launcher supports:

- one-cell startup from GitHub;
- automatic Drive mount, secret loading, dependency preparation, and Gradio launch;
- GPU transcription mode when T4 is available;
- CPU maintenance mode for loading existing transcripts, re-recognition, registry operations, and manual identity confirmation;
- temporary external UI access through a runtime magic-link that establishes an HttpOnly Secure session cookie without a separate username/password form;
- automatic browser-open attempt from the START cell with a single-button fallback when popups are blocked;
- a unified primary Process / Continue flow that opens an existing canonical transcript without retranscription and only starts WhisperX when no canonical transcript exists.

Operational rule: release the Colab runtime immediately after a full transcription is safely persisted to Drive. Do not leave T4 idle.

## 8. Current MVP completion status

The core end-to-end MVP acceptance path is now exercised:

1. fresh Colab worker starts from GitHub;
2. an existing Drive recording can be selected;
3. WhisperX transcription runs without manual code entry;
4. diarization and speaker IDs are produced;
5. previously registered speakers are automatically resolved when confidence is high;
6. uncertain speakers remain unknown instead of being misassigned;
7. only meaningful unresolved speakers are presented for naming;
8. manual linking updates the registry and regenerates JSON/TXT/DOCX without retranscription;
9. output files persist beside the source recording;
10. a subsequent recording has successfully reused the persistent speaker registry.

The principal remaining work is hardening and additional real-world validation, not filling a missing core MVP capability.

## 9. Important current limitations

1. Thresholds are based on a small real-world validation set and are not biometric-grade.
2. pyannote diarization can create short or noisy extra IDs and can mis-handle very short turn boundaries.
3. Very short incidental voices may be present in the transcript but should not be promoted to persistent identities without enough usable speech.
4. CPU re-recognition is a maintenance fallback and can be slower than GPU processing.
5. Voice embeddings remain sensitive data and stay only in the user's Drive registry.
6. Colab GPU quota and temporary tunnel reliability are operational constraints outside the core transcription logic.

## 10. Current automation direction

Keep the MVP single-file. Do not add multi-file transcription, a persistent job queue, or automatic Drive folder watching.

Automation work should remove operational steps around one selected recording. Priority order:

1. unified state-aware Process / Continue flow;
2. validate the automatic magic-link Gradio entry path in a fresh Colab/browser session;
3. automatic preflight after file selection and protection against accidental retranscription;
4. better representative audio previews for meaningful unknown speakers;
5. automatic finalization after the last meaningful unknown speaker is resolved;
6. one-action runtime/GPU release after completion;
7. benchmark startup/model caching only if measured startup cost justifies it.

Recognition thresholds remain conservative. Continue collecting real-world scores, margins, chunk counts, false negatives, and any false positives before changing the matching policy.

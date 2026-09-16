# Interview Transcriber — Project Status Snapshot

**Snapshot date:** 2026-09-17  
**Repository:** `playply/transcriber`  
**Current main:** `c7d7e7c530a64ff60ded79c451779c14323a15e3`

## 1. MVP goal

Build the smallest reliable workflow that takes an interview recording already stored on Google Drive, transcribes it in Russian, diarizes speakers, recognizes previously known speakers from a persistent registry, asks the user to name only unresolved speakers, and saves canonical JSON plus TXT and DOCX beside the source recording.

Out of scope for the MVP: permanent hosting, Telegram integration, native Mac/iOS apps, paid transcription SaaS, and a large dashboard.

## 2. Current stack

- Runtime: Google Colab, T4 when available.
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

## 4. What is already verified

### Core transcription vertical slice

A clean-room Colab run has been completed successfully from a fresh runtime. The system produced full Russian transcription, alignment, diarization, automatic speaker count, and JSON/TXT/DOCX outputs beside the source recording.

The clean-room test produced three concrete diarization speakers, 250 transcript segments, and only one residual `UNKNOWN` segment. This test does not need to be repeated unless the core transcription stack changes.

### Speaker embedding candidate

The WeSpeaker model is the current MVP embedding candidate. Initial within-recording validation showed strong separation between same-speaker and different-speaker embeddings, and the cross-recording test correctly matched a manually verified repeated person.

The current conservative automatic-match gates are:

- cosine similarity `>= 0.60`
- margin over runner-up `>= 0.20` when a runner-up exists
- at least `2` usable voice chunks

These values are deliberately conservative and are based on limited real interview data. They are not considered production-grade biometric thresholds.

### Persistent speaker registry

A persistent registry exists on Google Drive under the Interview Transcriber project area. It stores:

- stable `speaker_id`
- display name
- one or more voice embeddings
- source/provenance records
- timestamps and quality information

A person can accumulate multiple voice references across recordings. Diarization IDs such as `SPEAKER_00` or `SPEAKER_02` are recording-local labels and are not treated as identities.

### Automatic known-speaker resolution

The ordinary transcription flow already performs registry lookup after transcription and rewrites the canonical transcript with confident resolved names. TXT and DOCX are regenerated from the canonical JSON without changing the source diarization IDs.

A real cross-recording test has already demonstrated automatic resolution of a previously registered person while another speaker remained unresolved.

### Unknown-speaker confirmation

The Gradio UI supports the post-transcription flow for unresolved speakers:

1. show only unresolved diarization IDs;
2. show a representative timestamp/text snippet;
3. let the user either link the speaker to an existing registry identity or create a new identity;
4. save a usable voice reference when enough clean speech exists;
5. still allow transcript naming when there is too little speech for a safe reusable voice embedding;
6. regenerate JSON/TXT/DOCX without retranscription.

Identity linking is explicit by stable `speaker_id`; matching display-name text is no longer the identity key.

### Split diarization handling

A real recording demonstrated that pyannote can split one real person into more than one diarization ID in the same recording. The matcher was therefore changed so `unique_best` is diagnostic only and is no longer a hard gate.

Each diarization ID is now independently eligible for the same known identity when it passes the core gates (similarity, runner-up margin, and minimum usable speech). This allows two `SPEAKER_XX` IDs to resolve to one person when diarization has split that person's speech.

### Existing-transcript re-recognition

The UI now has `Re-recognize known speakers` for recordings that already have a canonical transcript. It recalculates speaker embeddings against the current registry, preserves prior user-confirmed assignments, updates confident automatic matches, and regenerates JSON/TXT/DOCX without rerunning WhisperX or diarization.

This is useful when the registry has learned additional people after an older recording was originally transcribed.

### CPU maintenance mode

The Colab launcher can start the Gradio UI even when Colab refuses GPU access because of quota limits. In CPU maintenance mode the user can work with existing transcripts and the speaker registry. New full transcription remains intentionally disabled without GPU.

## 5. Current UX

The main UI supports:

- selecting a recording on Google Drive;
- `Transcribe` when GPU is available;
- `Load existing transcript`;
- `Re-recognize known speakers`;
- reviewing unresolved speakers;
- linking an unresolved diarization speaker to an existing person;
- creating a new person;
- regenerating JSON/TXT/DOCX;
- inspecting the speaker registry.

The UI displays the current git build hash so it is possible to verify which code revision is actually running in Colab.

## 6. Important current limitations

1. **Fresh unseen-recording validation after the latest matcher change is still pending.** Existing `CONFIRMED` assignments with `score=n/a` do not prove automatic matching quality.
2. **Colab GPU quota is currently the blocker for the next clean end-to-end test.** Google does not provide a reliable reset timer.
3. **The current thresholds are based on a small validation set.** They should not be loosened merely to increase recall; changes must be driven by real false-negative/false-positive observations.
4. **Very short speakers may have no reusable voice embedding.** They can still be named in the current transcript, but they should not be treated as automatically recognizable until enough clean speech is captured later.
5. **CPU re-recognition can be slow.** It is a maintenance fallback, not the intended primary runtime.
6. **Voice embeddings remain sensitive data.** They stay in the user's Google Drive registry and are not committed to GitHub.

## 7. Next test — first task for the next session

Use a completely new recording that has not been used to create or confirm registry references and that contains at least one person already present in the registry.

Run only the ordinary `Transcribe` flow. Do not manually link or name speakers before inspecting the automatic result.

For every diarization ID record:

- `KNOWN` or `UNKNOWN`
- matched identity (if any)
- cosine score
- runner-up margin
- usable chunk count
- reason for acceptance/rejection
- manual ground-truth identity after listening

Success criterion for the next test: at least one previously known person is automatically resolved on the unseen recording with a real score/margin (not `CONFIRMED`, not `score=n/a`) and no incorrect automatic identity assignments occur.

Only after this test should similarity/margin thresholds or reference-management strategy be reconsidered.

## 8. Current repository milestones

Recent important revisions:

- `0834b96` — explicit linking of unknown speakers to stable existing identities.
- `6722e76` — split-diarization-aware known-speaker resolution while keeping the conservative core gates.
- `c7d7e7c` — re-recognition of known speakers on existing transcripts from the current registry.

## 9. MVP completion status

The transcription, diarization, persistent registry, manual unknown-speaker confirmation, canonical transcript rewriting, and repeat-use UX are implemented and individually exercised.

The principal remaining validation item before calling the MVP end-to-end complete is a clean automatic-recognition test on a fresh unseen recording using the current registry and the current split-diarization-aware matcher.

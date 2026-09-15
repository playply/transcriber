"""Speaker registry placeholder for the next MVP milestone.

Persistent speaker recognition is intentionally not implemented in the clean-room
transcription baseline. This module exists now only to reserve the agreed module
boundary without introducing embeddings, matching thresholds, or registry writes.
"""


class SpeakerRegistryNotImplemented(RuntimeError):
    pass


def load_registry(*args, **kwargs):
    raise SpeakerRegistryNotImplemented(
        "Speaker recognition is outside the clean-room transcription baseline."
    )

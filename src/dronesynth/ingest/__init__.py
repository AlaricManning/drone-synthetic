"""Run registration and upload.

Validates a completed EasySynth capture (normal/mask pairing, frame counts),
writes its manifest, and syncs it to raw storage. Frames upload first,
manifest last: a run without a manifest is incomplete by definition.
"""

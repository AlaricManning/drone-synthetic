"""Storage abstraction over local paths and S3.

Ingest and datagen read/write through this layer so the same code runs
against local staging directories in development and s3:// URIs in
production.
"""

from dronesynth.storage.backends import (
    LocalStorage,
    S3Storage,
    Storage,
    StorageError,
    storage_for,
)

__all__ = ["LocalStorage", "S3Storage", "Storage", "StorageError", "storage_for"]

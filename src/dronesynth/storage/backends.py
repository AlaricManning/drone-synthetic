"""Storage backends: the same operations over a local directory or an S3 prefix.

A storage *root* is a string from config: either a filesystem path
(``data/raw``) or an S3 URI (``s3://bucket/raw``). ``storage_for`` picks the
backend; everything above it addresses files by *key* — a relative,
``/``-separated path like ``run_0001/manifest.json`` — and never knows which
backend it is talking to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from shutil import copy2, rmtree


class StorageError(Exception):
    """Raised when a storage operation fails in a way we can name."""


class StorageKeyExists(StorageError):
    """Raised by write_text_if_absent when the key is already there."""


class StorageKeyMissing(StorageError):
    """Raised by read_text/get_file when the key does not exist."""


class StorageNotPermitted(StorageError):
    """Raised when the current credentials cannot perform the operation."""


class Storage(ABC):
    """put/get/list/exists over keys under a root."""

    @abstractmethod
    def put_file(self, source: Path, key: str) -> None: ...

    @abstractmethod
    def get_file(self, key: str, dest: Path) -> None: ...

    @abstractmethod
    def write_text(self, key: str, text: str) -> None: ...

    @abstractmethod
    def read_text(self, key: str) -> str: ...

    @abstractmethod
    def write_text_if_absent(self, key: str, text: str) -> None:
        """Write only if the key doesn't exist; raise StorageKeyExists if it does.

        This is the primitive behind run immutability: the manifest is written
        with if-absent semantics, so two ingests of the same run id cannot both
        succeed — even over S3, where a put-only identity cannot check
        existence beforehand.
        """

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]: ...

    @abstractmethod
    def delete_prefix(self, prefix: str) -> None:
        """Remove everything under a prefix; StorageNotPermitted if impossible.

        Local storage supports this (clearing debris from a failed ingest);
        the S3 ingest identity deliberately cannot delete, so callers must
        treat this as best-effort.
        """

    @abstractmethod
    def describe(self, key: str = "") -> str:
        """Human-readable location of a key, for logs and errors."""


class LocalStorage(Storage):
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key

    def put_file(self, source: Path, key: str) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, dest)

    def get_file(self, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            copy2(self._path(key), dest)
        except FileNotFoundError as exc:
            raise StorageKeyMissing(f"{self.describe(key)} does not exist") from exc

    def write_text(self, key: str, text: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def read_text(self, key: str) -> str:
        try:
            return self._path(key).read_text()
        except FileNotFoundError as exc:
            raise StorageKeyMissing(f"{self.describe(key)} does not exist") from exc

    def write_text_if_absent(self, key: str, text: str) -> None:
        path = self._path(key)
        if path.exists():
            raise StorageKeyExists(f"{self.describe(key)} already exists")
        self.write_text(key, text)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete_prefix(self, prefix: str) -> None:
        base = self._path(prefix)
        if base.exists():
            rmtree(base)

    def list_keys(self, prefix: str = "") -> list[str]:
        base = self._path(prefix) if prefix else self.root
        if not base.is_dir():
            return []
        return sorted(
            str(p.relative_to(self.root).as_posix()) for p in base.rglob("*") if p.is_file()
        )

    def describe(self, key: str = "") -> str:
        return str(self._path(key)) if key else str(self.root)


class S3Storage(Storage):
    def __init__(self, bucket: str, prefix: str = "", client=None) -> None:
        if client is None:
            import boto3

            client = boto3.client("s3")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = client

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_file(self, source: Path, key: str) -> None:
        self.client.upload_file(str(source), self.bucket, self._key(key))

    def get_file(self, key: str, dest: Path) -> None:
        from botocore.exceptions import ClientError

        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.download_file(self.bucket, self._key(key), str(dest))
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                raise StorageKeyMissing(f"{self.describe(key)} does not exist") from exc
            raise

    def write_text(self, key: str, text: str) -> None:
        self.client.put_object(
            Bucket=self.bucket, Key=self._key(key), Body=text.encode("utf-8")
        )

    def read_text(self, key: str) -> str:
        from botocore.exceptions import ClientError

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                raise StorageKeyMissing(f"{self.describe(key)} does not exist") from exc
            raise
        return response["Body"].read().decode("utf-8")

    def write_text_if_absent(self, key: str, text: str) -> None:
        from botocore.exceptions import ClientError

        try:
            # S3 conditional write: fails with 412 if the object already exists
            self.client.put_object(
                Bucket=self.bucket,
                Key=self._key(key),
                Body=text.encode("utf-8"),
                IfNoneMatch="*",
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("PreconditionFailed", "412"):
                raise StorageKeyExists(f"{self.describe(key)} already exists") from exc
            raise

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            if code in ("403", "AccessDenied"):
                raise StorageNotPermitted(
                    f"cannot check {self.describe(key)}: credentials lack read access"
                ) from exc
            raise
        return True

    def delete_prefix(self, prefix: str) -> None:
        raise StorageNotPermitted(
            "the ingest identity is put-only by design; clean up S3 debris with "
            "admin credentials if needed"
        )

    def list_keys(self, prefix: str = "") -> list[str]:
        full_prefix = self._key(prefix) if prefix else self.prefix
        if full_prefix and not full_prefix.endswith("/"):
            full_prefix += "/"
        keys = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for entry in page.get("Contents", []):
                key = entry["Key"]
                if self.prefix:
                    key = key[len(self.prefix) + 1 :]
                keys.append(key)
        return sorted(keys)

    def describe(self, key: str = "") -> str:
        suffix = f"/{self._key(key)}" if key else (f"/{self.prefix}" if self.prefix else "")
        return f"s3://{self.bucket}{suffix}"


def storage_for(root: str, client=None) -> Storage:
    """Pick a backend from a config root: ``s3://bucket/prefix`` or a local path."""
    if root.startswith("s3://"):
        without_scheme = root[len("s3://") :]
        bucket, _, prefix = without_scheme.partition("/")
        if not bucket:
            raise StorageError(f"invalid S3 root {root!r}: no bucket name")
        return S3Storage(bucket=bucket, prefix=prefix, client=client)
    return LocalStorage(Path(root))

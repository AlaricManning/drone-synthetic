import boto3
import pytest
from moto import mock_aws

from dronesynth.storage import (
    LocalStorage,
    S3Storage,
    StorageError,
    StorageKeyExists,
    StorageKeyMissing,
    StorageNotPermitted,
    storage_for,
)
from dronesynth.storage.backends import CONNECTION_POOL_SIZE


@pytest.fixture(params=["local", "s3"])
def storage(request, tmp_path):
    """Both backends must satisfy the same contract, so every test runs on both."""
    if request.param == "local":
        yield LocalStorage(tmp_path / "root")
    else:
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bucket")
            yield S3Storage(bucket="test-bucket", prefix="raw", client=client)


@pytest.fixture
def s3_client():
    """A live mock bucket, for the copy tests that need two roots at once."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="test-bucket")
        yield client


def test_put_and_get_file(storage, tmp_path):
    source = tmp_path / "in.txt"
    source.write_text("payload")
    storage.put_file(source, "run_0001/normal/frame_000000.png")

    dest = tmp_path / "deep" / "out.txt"
    storage.get_file("run_0001/normal/frame_000000.png", dest)
    assert dest.read_text() == "payload"


def test_text_round_trip(storage):
    storage.write_text("run_0001/manifest.json", '{"run_id": "run_0001"}')
    assert storage.read_text("run_0001/manifest.json") == '{"run_id": "run_0001"}'


def test_exists(storage):
    assert not storage.exists("run_0001/manifest.json")
    storage.write_text("run_0001/manifest.json", "{}")
    assert storage.exists("run_0001/manifest.json")


def test_list_keys(storage):
    storage.write_text("run_0001/manifest.json", "{}")
    storage.write_text("run_0001/normal/frame_000000.png", "x")
    storage.write_text("run_0002/manifest.json", "{}")

    assert storage.list_keys("run_0001") == [
        "run_0001/manifest.json",
        "run_0001/normal/frame_000000.png",
    ]
    assert len(storage.list_keys()) == 3
    assert storage.list_keys("run_0009") == []


def test_write_text_if_absent(storage):
    storage.write_text_if_absent("run_0001/manifest.json", "first")
    with pytest.raises(StorageKeyExists, match="already exists"):
        storage.write_text_if_absent("run_0001/manifest.json", "second")
    assert storage.read_text("run_0001/manifest.json") == "first"


def test_copy_from_within_one_root(storage):
    storage.write_text("run_0001/normal/frame_000000.png", "pixels")
    storage.copy_from(storage, "run_0001/normal/frame_000000.png", "v001/images/train/a.png")

    assert storage.read_text("v001/images/train/a.png") == "pixels"
    # A copy, not a move.
    assert storage.exists("run_0001/normal/frame_000000.png")


def test_copy_from_missing_source_key(storage):
    with pytest.raises(StorageKeyMissing, match="does not exist"):
        storage.copy_from(storage, "run_0009/gone.png", "v001/images/train/a.png")


def test_copy_from_across_roots_in_one_bucket(s3_client):
    """The production pairing: raw and datasets are separate roots, same bucket."""
    raw = S3Storage(bucket="test-bucket", prefix="raw", client=s3_client)
    datasets = S3Storage(bucket="test-bucket", prefix="datasets", client=s3_client)
    raw.write_text("run_0001/normal/frame_000000.png", "pixels")

    datasets.copy_from(raw, "run_0001/normal/frame_000000.png", "v001/yolo/images/train/a.png")

    assert datasets.read_text("v001/yolo/images/train/a.png") == "pixels"
    assert raw.exists("run_0001/normal/frame_000000.png")


def test_copy_from_local_to_s3(s3_client, tmp_path):
    local = LocalStorage(tmp_path / "raw")
    remote = S3Storage(bucket="test-bucket", prefix="datasets", client=s3_client)
    local.write_text("run_0001/normal/frame_000000.png", "pixels")

    remote.copy_from(local, "run_0001/normal/frame_000000.png", "v001/images/train/a.png")

    assert remote.read_text("v001/images/train/a.png") == "pixels"


def test_copy_from_s3_to_local(s3_client, tmp_path):
    remote = S3Storage(bucket="test-bucket", prefix="raw", client=s3_client)
    local = LocalStorage(tmp_path / "datasets")
    remote.write_text("run_0001/normal/frame_000000.png", "pixels")

    local.copy_from(remote, "run_0001/normal/frame_000000.png", "v001/images/train/a.png")

    assert local.read_text("v001/images/train/a.png") == "pixels"


def test_delete_prefix_local(tmp_path):
    storage = LocalStorage(tmp_path / "root")
    storage.write_text("run_0001/normal/frame.png", "x")
    storage.write_text("run_0002/manifest.json", "{}")
    storage.delete_prefix("run_0001")
    assert storage.list_keys() == ["run_0002/manifest.json"]
    storage.delete_prefix("run_0009")  # deleting nothing is fine


def test_delete_prefix_s3_not_permitted():
    storage = S3Storage(bucket="b", prefix="raw", client=object())
    with pytest.raises(StorageNotPermitted, match="put-only"):
        storage.delete_prefix("run_0001")


def test_storage_for_local(tmp_path):
    storage = storage_for(str(tmp_path / "data" / "raw"))
    assert isinstance(storage, LocalStorage)


def test_storage_for_s3():
    storage = storage_for("s3://my-bucket/raw", client=object())
    assert isinstance(storage, S3Storage)
    assert storage.bucket == "my-bucket"
    assert storage.prefix == "raw"
    assert storage.describe("run_0001/manifest.json") == (
        "s3://my-bucket/raw/run_0001/manifest.json"
    )


def test_the_connection_pool_holds_every_concurrent_caller():
    """A pool smaller than the caller's thread count costs a handshake per request.

    botocore does not queue for a free connection: it opens an extra one and
    discards it on release, so an undersized pool turns concurrency into a
    stream of TLS setups. The build is the concurrent caller today.
    """
    from dronesynth.datagen.build import PLACEMENT_WORKERS

    assert PLACEMENT_WORKERS <= CONNECTION_POOL_SIZE

    with mock_aws():
        storage = S3Storage(bucket="pool-check")
        assert storage.client.meta.config.max_pool_connections == CONNECTION_POOL_SIZE


def test_storage_for_s3_without_bucket_rejected():
    with pytest.raises(StorageError, match="no bucket"):
        storage_for("s3://")

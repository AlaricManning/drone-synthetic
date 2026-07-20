import boto3
import pytest
from moto import mock_aws

from dronesynth.storage import LocalStorage, S3Storage, StorageError, storage_for


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


def test_storage_for_s3_without_bucket_rejected():
    with pytest.raises(StorageError, match="no bucket"):
        storage_for("s3://")

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, mock_open, patch

import pytest

from app.services import gcs_service


class FakeBlob:
    def __init__(self, name, updated=None, exists=True, text="{}"):
        self.name = name
        self.updated = updated or datetime.now(UTC)
        self._exists = exists
        self._text = text
        self.upload_from_file = MagicMock()
        self.upload_from_filename = MagicMock()
        self.upload_from_string = MagicMock()
        self.download_to_filename = MagicMock()

    def exists(self):
        return self._exists

    def download_as_text(self):
        return self._text


class FakeBucket:
    def __init__(self, blob=None):
        self._blob = blob or FakeBlob("dummy")
        self.blob = MagicMock(return_value=self._blob)


class FakeClient:
    def __init__(self, bucket=None, blobs=None):
        self._bucket = bucket or FakeBucket()
        self._blobs = blobs or []
        self.bucket = MagicMock(return_value=self._bucket)
        self.list_blobs = MagicMock(return_value=self._blobs)


def test_get_client_returns_storage_client():
    with patch.object(gcs_service.storage, "Client") as mock_client:
        expected = MagicMock()
        mock_client.return_value = expected

        result = gcs_service.get_client()

        assert result is expected
        mock_client.assert_called_once_with()


def test_upload_file_to_gcs_uses_upload_from_file_for_small_files(tmp_path, monkeypatch):
    file_path = tmp_path / "small.txt"
    file_path.write_text("hello")

    blob = FakeBlob("target")
    bucket = FakeBucket(blob)
    client = FakeClient(bucket=bucket)
    monkeypatch.setattr(gcs_service, "get_client", lambda: client)
    monkeypatch.setattr(gcs_service, "_MULTIPART_THRESHOLD", 1024 * 1024)

    with patch("builtins.open", mock_open(read_data="hello")) as mocked_open:
        gcs_service.upload_file_to_gcs(
            bucket_name="bucket",
            local_path=str(file_path),
            blob_name="blob.txt",
            max_attempts=1,
        )

    client.bucket.assert_called_once_with("bucket")
    bucket.blob.assert_called_once_with("blob.txt")
    blob.upload_from_file.assert_called_once()
    blob.upload_from_filename.assert_not_called()
    mocked_open.assert_called_once_with(str(file_path), "rb")


def test_upload_file_to_gcs_uses_upload_from_filename_for_large_files(tmp_path, monkeypatch):
    file_path = tmp_path / "large.bin"
    file_path.write_bytes(b"x" * 10)

    blob = FakeBlob("target")
    bucket = FakeBucket(blob)
    client = FakeClient(bucket=bucket)
    monkeypatch.setattr(gcs_service, "get_client", lambda: client)
    monkeypatch.setattr(gcs_service, "_MULTIPART_THRESHOLD", 1)

    gcs_service.upload_file_to_gcs(
        bucket_name="bucket",
        local_path=str(file_path),
        blob_name="blob.bin",
        max_attempts=1,
    )

    blob.upload_from_filename.assert_called_once_with(str(file_path), retry=gcs_service._RETRY)
    blob.upload_from_file.assert_not_called()


def test_upload_file_to_gcs_retries_then_raises(tmp_path, monkeypatch):
    file_path = tmp_path / "file.txt"
    file_path.write_text("hello")

    blob = FakeBlob("target")
    blob.upload_from_file.side_effect = Exception("temporary failure")
    bucket = FakeBucket(blob)
    client = FakeClient(bucket=bucket)
    monkeypatch.setattr(gcs_service, "get_client", lambda: client)
    monkeypatch.setattr(gcs_service, "_MULTIPART_THRESHOLD", 1024 * 1024)

    with patch("time.sleep") as sleep_mock, pytest.raises(
        RuntimeError,
        match="Upload GCS abandonné",
    ):
        gcs_service.upload_file_to_gcs(
            bucket_name="bucket",
            local_path=str(file_path),
            blob_name="blob.txt",
            max_attempts=2,
        )

    assert blob.upload_from_file.call_count == 2
    sleep_mock.assert_called_once_with(2)


def test_upload_json_to_gcs(monkeypatch):
    blob = FakeBlob("json")
    bucket = FakeBucket(blob)
    client = FakeClient(bucket=bucket)
    monkeypatch.setattr(gcs_service, "get_client", lambda: client)

    data = {"name": "Signal-Conso", "ok": True}
    gcs_service.upload_json_to_gcs("bucket", "data.json", data)

    blob.upload_from_string.assert_called_once()
    args, kwargs = blob.upload_from_string.call_args
    assert json.loads(args[0]) == data
    assert kwargs["content_type"] == "application/json"
    assert kwargs["retry"] == gcs_service._RETRY


def test_download_json_from_gcs_returns_none_when_missing(monkeypatch):
    blob = FakeBlob("json", exists=False)
    bucket = FakeBucket(blob)
    client = FakeClient(bucket=bucket)
    monkeypatch.setattr(gcs_service, "get_client", lambda: client)

    result = gcs_service.download_json_from_gcs("bucket", "missing.json")

    assert result is None
    blob.download_as_text.assert_not_called()


def test_download_json_from_gcs_returns_parsed_json(monkeypatch):
    payload = {"id": "123", "score": 0.91}
    blob = FakeBlob("json", exists=True, text=json.dumps(payload))
    bucket = FakeBucket(blob)
    client = FakeClient(bucket=bucket)
    monkeypatch.setattr(gcs_service, "get_client", lambda: client)

    result = gcs_service.download_json_from_gcs("bucket", "data.json")

    assert result == payload
    blob.download_as_text.assert_called_once()


def test_find_prediction_in_bucket(monkeypatch):
    target_payload = {"id": "pred-42", "label": "spam"}
    blobs = [
        FakeBlob("predictions/2024-01-01.json", text=json.dumps({"id": "x"})),
        FakeBlob("predictions/pred-42.json", text=json.dumps(target_payload)),
    ]
    client = FakeClient(blobs=blobs)
    monkeypatch.setattr(gcs_service, "get_client", lambda: client)

    result = gcs_service.find_prediction_in_bucket("bucket", "pred-42")

    assert result == target_payload


def test_get_latest_blob_returns_latest_name(monkeypatch):
    blobs = [
        FakeBlob("data/a.json", updated=datetime(2024, 1, 1, tzinfo=UTC)),
        FakeBlob("data/b.json", updated=datetime(2024, 2, 1, tzinfo=UTC)),
    ]
    client = FakeClient(blobs=blobs)
    monkeypatch.setattr(gcs_service, "get_client", lambda: client)

    result = gcs_service.get_latest_blob("bucket", prefix="data/")

    assert result == "data/b.json"


def test_get_latest_blob_returns_none_when_empty(monkeypatch):
    client = FakeClient(blobs=[])
    monkeypatch.setattr(gcs_service, "get_client", lambda: client)

    result = gcs_service.get_latest_blob("bucket", prefix="data/")

    assert result is None


def test_download_blob_to_file(monkeypatch):
    blob = FakeBlob("file.txt")
    bucket = FakeBucket(blob)
    client = FakeClient(bucket=bucket)
    monkeypatch.setattr(gcs_service, "get_client", lambda: client)

    gcs_service.download_blob_to_file("bucket", "file.txt", "dest.txt")

    bucket.blob.assert_called_once_with("file.txt")
    blob.download_to_filename.assert_called_once_with("dest.txt", retry=gcs_service._RETRY)

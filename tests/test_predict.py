from unittest.mock import MagicMock

import numpy as np
import pytest

from app.ml import predict as predict_module
from app.ml.predict import TicketModel, normalize_text


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, ""),
        ("  Héllo WORLD  ", "hello world"),
        ("Contact: TEST@Example.com", "contact"),
        ("Voir https://example.com/page", "voir"),
        ("Bonjour!!!   ça va?", "bonjour ca va"),
    ],
)
def test_normalize_text(value, expected):
    assert normalize_text(value) == expected


def test_download_from_gcs_raises_without_bucket_or_blob(tmp_path):
    model = TicketModel(model_path=str(tmp_path / "model.joblib"))

    with pytest.raises(FileNotFoundError, match="Modèle introuvable localement"):
        model.download_from_gcs()


def test_download_from_gcs_raises_when_blob_missing(monkeypatch, tmp_path):
    fake_blob = MagicMock()
    fake_blob.exists.return_value = False

    fake_bucket = MagicMock()
    fake_bucket.blob.return_value = fake_blob

    fake_client = MagicMock()
    fake_client.bucket.return_value = fake_bucket

    monkeypatch.setattr(predict_module.storage, "Client", MagicMock(return_value=fake_client))

    model = TicketModel(
        model_path=str(tmp_path / "model.joblib"),
        bucket_name="bucket",
        blob_name="models/model.joblib",
    )

    with pytest.raises(FileNotFoundError, match="Modèle introuvable dans GCS"):
        model.download_from_gcs()

    fake_client.bucket.assert_called_once_with("bucket")
    fake_bucket.blob.assert_called_once_with("models/model.joblib")
    fake_blob.exists.assert_called_once()


def test_download_from_gcs_downloads_file(monkeypatch, tmp_path):
    fake_blob = MagicMock()
    fake_blob.exists.return_value = True

    fake_bucket = MagicMock()
    fake_bucket.blob.return_value = fake_blob

    fake_client = MagicMock()
    fake_client.bucket.return_value = fake_bucket

    monkeypatch.setattr(predict_module.storage, "Client", MagicMock(return_value=fake_client))

    model_path = tmp_path / "nested" / "model.joblib"
    model = TicketModel(
        model_path=str(model_path),
        bucket_name="bucket",
        blob_name="models/model.joblib",
    )

    model.download_from_gcs()

    assert model_path.parent.exists()
    fake_blob.download_to_filename.assert_called_once_with(str(model_path))


def test_load_uses_local_file(monkeypatch, tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"dummy")

    fake_loaded_model = MagicMock()
    load_mock = MagicMock(return_value=fake_loaded_model)
    monkeypatch.setattr(predict_module.joblib, "load", load_mock)

    model = TicketModel(model_path=str(model_path))
    model.load()

    assert model.model is fake_loaded_model
    load_mock.assert_called_once()
    assert load_mock.call_args.args[0] == model_path


def test_load_downloads_when_missing(monkeypatch, tmp_path):
    model_path = tmp_path / "model.joblib"
    loaded_model = MagicMock()

    def fake_download(self):
        path = Path(self.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"downloaded")

    from pathlib import Path

    monkeypatch.setattr(TicketModel, "download_from_gcs", fake_download)
    monkeypatch.setattr(predict_module.joblib, "load", MagicMock(return_value=loaded_model))

    model = TicketModel(
        model_path=str(model_path),
        bucket_name="bucket",
        blob_name="models/model.joblib",
    )

    model.load()

    assert model.model is loaded_model
    assert model_path.exists()


def test_load_raises_when_still_missing_after_download(monkeypatch, tmp_path):
    model_path = tmp_path / "model.joblib"

    monkeypatch.setattr(TicketModel, "download_from_gcs", lambda self: None)

    model = TicketModel(model_path=str(model_path))

    with pytest.raises(FileNotFoundError, match="Modèle introuvable"):
        model.load()


def test_predict_lazy_loads_and_returns_label(monkeypatch):
    model = TicketModel(model_path="model.joblib")
    fake_model = MagicMock()
    fake_model.predict.return_value = ["cat"]

    monkeypatch.setattr(
        model, "load", MagicMock(side_effect=lambda: setattr(model, "model", fake_model))
    )

    result = model.predict("  Bonjour le monde !!! ")

    assert result == "cat"
    fake_model.predict.assert_called_once_with(["bonjour le monde"])


def test_predict_raises_on_empty_text(monkeypatch):
    model = TicketModel(model_path="model.joblib")
    model.model = MagicMock()

    with pytest.raises(ValueError, match="Le texte d'entrée est vide ou invalide"):
        model.predict("   ")


def test_predict_many_filters_invalid_texts(monkeypatch):
    model = TicketModel(model_path="model.joblib")
    fake_model = MagicMock()
    fake_model.predict.return_value = ["a", "b"]

    monkeypatch.setattr(
        model, "load", MagicMock(side_effect=lambda: setattr(model, "model", fake_model))
    )

    result = model.predict_many([" Hello ", None, "", "World!!", "   "])

    assert result == ["a", "b"]
    fake_model.predict.assert_called_once_with(["hello", "world"])


def test_predict_many_raises_when_no_valid_text(monkeypatch):
    model = TicketModel(model_path="model.joblib")
    model.model = MagicMock()

    with pytest.raises(ValueError, match="Aucun texte valide à prédire"):
        model.predict_many(["", None, "   "])


def test_predict_with_proba_returns_label_and_score(monkeypatch):
    model = TicketModel(model_path="model.joblib")
    fake_model = MagicMock()
    fake_model.predict_proba.return_value = np.array([[0.1, 0.9]])
    fake_model.classes_ = np.array(["low", "high"])

    monkeypatch.setattr(
        model, "load", MagicMock(side_effect=lambda: setattr(model, "model", fake_model))
    )

    label, score = model.predict_with_proba("Très bon produit !!!")

    assert label == "high"
    assert score == pytest.approx(0.9)
    fake_model.predict_proba.assert_called_once_with(["tres bon produit"])


def test_predict_with_proba_raises_on_empty_text(monkeypatch):
    model = TicketModel(model_path="model.joblib")
    model.model = MagicMock()

    with pytest.raises(ValueError, match="Le texte d'entrée est vide ou invalide"):
        model.predict_with_proba("   ")

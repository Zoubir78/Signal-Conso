from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.ml.train import (
    AVAILABLE_MODELS,
    build_pipeline,
    load_model,
    predict,
    train_model,
)


@pytest.fixture
def dummy_df():
    """Génère un DataFrame de test avec suffisamment de données."""
    data = {
        "clean_text": [f"ceci est un message de test numéro {i}" for i in range(100)],
        "category": ["A" if i < 50 else "B" for i in range(100)],
    }
    return pd.DataFrame(data)


# --- Tests pour build_pipeline ---


def test_build_pipeline_valid():
    """Vérifie que la construction du pipeline fonctionne pour un modèle valide."""
    pipeline = build_pipeline("logreg")
    assert "tfidf" in pipeline.named_steps
    assert "clf" in pipeline.named_steps
    # Correction E721 : Utilisation de isinstance avec le type attendu
    assert isinstance(pipeline.named_steps["clf"], type(AVAILABLE_MODELS["logreg"]))


def test_build_pipeline_invalid():
    """Vérifie qu'une erreur est levée pour un modèle inconnu."""
    with pytest.raises(ValueError, match="Modèle inconnu"):
        build_pipeline("modele_imaginaire")


# --- Tests pour train_model ---


@patch("app.ml.train.joblib.dump")
@patch("app.ml.train.Path.mkdir")
def test_train_model_success(mock_mkdir, mock_dump, dummy_df):
    """Teste le cycle complet d'entraînement avec succès."""
    results = train_model(
        df=dummy_df,
        model_name="logreg",
        model_path="fake_models/model.joblib",
        min_class_samples=5,
    )

    assert "accuracy" in results
    assert "f1_macro" in results
    assert results["n_classes"] == 2
    assert results["n_train"] == 80  # 80% de 100 par défaut
    assert results["model_path"] == "fake_models/model.joblib"

    # Vérifie que la sauvegarde a été appelée
    mock_dump.assert_called_once()
    # Vérifie que la création du dossier a été tentée
    mock_mkdir.assert_called()


def test_train_model_missing_column():
    """Vérifie l'erreur si une colonne est manquante."""
    df_invalid = pd.DataFrame({"wrong_col": ["text"]})
    with pytest.raises(ValueError, match="Colonne absente"):
        train_model(df=df_invalid)


def test_train_model_too_small():
    """Vérifie l'erreur si le dataset est trop petit après nettoyage."""
    df_small = pd.DataFrame({"clean_text": ["court"] * 10, "category": ["A"] * 10})
    with pytest.raises(ValueError, match="Jeu de données trop petit"):
        train_model(df=df_small, min_class_samples=1)


# --- Tests pour load_model ---


@patch("app.ml.train.Path.exists")
@patch("app.ml.train.joblib.load")
def test_load_model_success(mock_load, mock_exists):
    """Teste le chargement si le fichier existe."""
    mock_exists.return_value = True
    load_model("existing_model.joblib")
    mock_load.assert_called_once()


def test_load_model_not_found():
    """Teste l'erreur si le fichier n'existe pas."""
    with pytest.raises(FileNotFoundError):
        load_model("ghost.joblib")


# --- Tests pour predict ---


@patch("app.ml.train.load_model")
def test_predict_format(mock_load_model):
    """Vérifie que la fonction de prédiction renvoie le bon format de données."""
    # Simulation d'un modèle avec predict et predict_proba
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array(["A", "B"])
    # predict_proba renvoie des probabilités pour 2 classes
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.1, 0.9]])
    mock_load_model.return_value = mock_model

    test_texts = ["message 1", "message 2"]
    results = predict(test_texts)

    assert len(results) == 2
    assert results[0]["category"] == "A"
    assert results[0]["confidence"] == 0.8
    assert results[1]["category"] == "B"
    assert results[1]["confidence"] == 0.9

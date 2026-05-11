from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scripts.pipeline import (
    DBT_EXECUTABLE,
    DBT_PROFILES_DIR,
    DBT_TARGET,
    MODELS_TO_TRAIN,
    _find_dbt,
    _print_leaderboard,
    _run_dbt,
    run_pipeline,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_settings():
    with patch("scripts.pipeline.get_settings") as mock:
        settings = MagicMock()
        settings.GCS_BUCKET_NAME = "test-bucket"
        settings.GCP_PROJECT_ID = "test-project"
        mock.return_value = settings
        yield mock


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame(
        {
            "id": [1, 2],
            "text": ["bon produit", "service médiocre"],
            "category": ["alimentaire", "service"],
        }
    )


@pytest.fixture
def sample_mart_df():
    return pd.DataFrame(
        {
            "clean_text": ["bon produit", "service médiocre"],
            "category": ["alimentaire", "service"],
            "is_valid": [True, True],
        }
    )


@pytest.fixture
def mock_train_metrics():
    def _metrics(model_name):
        return {
            "model_name": model_name,
            "accuracy": 0.85 if model_name == "logreg" else 0.80,
            "f1_macro": 0.84 if model_name == "logreg" else 0.79,
            "n_train": 800,
            "n_test": 200,
            "n_classes": 3,
        }

    return _metrics


# -----------------------------------------------------------------------------
# Tests pour _find_dbt
# -----------------------------------------------------------------------------


@patch("scripts.pipeline.Path.exists")
@patch("scripts.pipeline.shutil.which")
@patch("scripts.pipeline.sys.platform", "linux")
def test_find_dbt_venv(mock_which, mock_exists):
    """Retourne l'exécutable dans le venv s'il existe."""
    mock_which.return_value = None
    # Simuler venv/dbt existant
    with patch("scripts.pipeline.Path") as mock_path:
        mock_venv_dbt = MagicMock()
        mock_venv_dbt.exists.return_value = True
        mock_path.return_value.parent.return_value.__truediv__.return_value = mock_venv_dbt
        # Forcer le chemin du venv
        with patch("scripts.pipeline.sys.executable", "/venv/bin/python"):
            result = _find_dbt()
            assert result.endswith("dbt")


@patch("scripts.pipeline.Path.exists")
@patch("scripts.pipeline.shutil.which")
@patch("scripts.pipeline.sys.platform", "linux")
def test_find_dbt_system_path(mock_which, mock_exists):
    """Retourne dbt trouvé dans PATH."""
    mock_which.return_value = "/usr/bin/dbt"
    # Aucun venv
    with patch("scripts.pipeline.Path") as mock_path:
        mock_venv_dbt = MagicMock()
        mock_venv_dbt.exists.return_value = False
        mock_path.return_value.parent.return_value.__truediv__.return_value = mock_venv_dbt
        result = _find_dbt()
        assert result == "/usr/bin/dbt"


@patch("scripts.pipeline.Path.exists")
@patch("scripts.pipeline.shutil.which")
@patch("scripts.pipeline.sys.platform", "linux")
def test_find_dbt_fallback_docker(mock_which, mock_exists):
    """Fallback sur /usr/local/bin/dbt si non trouvé ailleurs."""
    mock_which.return_value = None
    # Pas de venv
    with patch("scripts.pipeline.Path") as mock_path:
        mock_venv_dbt = MagicMock()
        mock_venv_dbt.exists.return_value = False
        mock_path.return_value.parent.return_value.__truediv__.return_value = mock_venv_dbt
        # Simuler l'existence de /usr/local/bin/dbt
        mock_path.side_effect = lambda p: MagicMock(exists=lambda: p == "/usr/local/bin/dbt")
        result = _find_dbt()
        assert result == "/usr/local/bin/dbt"


# -----------------------------------------------------------------------------
# Tests pour _run_dbt
# -----------------------------------------------------------------------------


@patch("scripts.pipeline.subprocess.run")
@patch("scripts.pipeline.TemporaryDirectory")
def test_run_dbt_success(mock_temp_dir, mock_subprocess_run):
    """Vérifie que _run_dbt appelle subprocess correctement et ne lève pas d'erreur."""
    # Simuler des répertoires temporaires
    mock_temp_dir.return_value.__enter__.return_value = "/tmp/dbt_logs"
    mock_temp_dir.side_effect = ["/tmp/dbt_logs", "/tmp/dbt_target"]

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Finished running"
    mock_result.stderr = ""
    mock_subprocess_run.return_value = mock_result

    log = MagicMock()

    _run_dbt(log, target="dev")

    # Vérifier les appels
    mock_subprocess_run.assert_called_once()
    args, kwargs = mock_subprocess_run.call_args
    cmd = args[0]
    assert cmd[0] == DBT_EXECUTABLE
    assert cmd[1] == "run"
    assert "--target" in cmd
    assert cmd[cmd.index("--target") + 1] == "dev"
    assert kwargs["cwd"] == str(Path(__file__).resolve().parents[2] / "dbt")
    assert kwargs["env"]["DBT_PROFILES_DIR"] == DBT_PROFILES_DIR
    log.assert_any_call("  ▶ Exécutable : " + DBT_EXECUTABLE)


@patch("scripts.pipeline.subprocess.run")
@patch("scripts.pipeline.TemporaryDirectory")
def test_run_dbt_failure(mock_temp_dir, mock_subprocess_run):
    """Lève RuntimeError si dbt run échoue."""
    mock_temp_dir.return_value.__enter__.return_value = "/tmp/dbt_logs"
    mock_temp_dir.side_effect = ["/tmp/dbt_logs", "/tmp/dbt_target"]

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = "Error"
    mock_result.stderr = "Something wrong"
    mock_subprocess_run.return_value = mock_result

    log = MagicMock()

    with pytest.raises(RuntimeError, match="dbt run a échoué"):
        _run_dbt(log)


# -----------------------------------------------------------------------------
# Tests pour _print_leaderboard
# -----------------------------------------------------------------------------


def test_print_leaderboard(capsys):
    """Vérifie l'affichage du leaderboard."""
    results = [
        {"model_name": "logreg", "accuracy": 0.85, "f1_macro": 0.84, "n_train": 800, "n_test": 200},
        {"model_name": "svc", "accuracy": 0.80, "f1_macro": 0.79, "n_train": 800, "n_test": 200},
    ]
    log = print  # simple print pour capturer
    _print_leaderboard(results, log)
    captured = capsys.readouterr()
    assert "logreg" in captured.out
    assert "85.00%" in captured.out
    assert "svc" in captured.out
    assert "80.00%" in captured.out
    assert "🏆" in captured.out  # le premier a la couronne


# -----------------------------------------------------------------------------
# Tests pour run_pipeline (intégration mockée)
# -----------------------------------------------------------------------------


@patch("scripts.pipeline.export_mart_to_gcs")
@patch("scripts.pipeline.upload_json_to_gcs")
@patch("scripts.pipeline.upload_file_to_gcs")
@patch("scripts.pipeline.train_model")
@patch("scripts.pipeline.read_mart_table")
@patch("scripts.pipeline._run_dbt")
@patch("scripts.pipeline.extract_from_signalconso_api")
@patch("scripts.pipeline.Path.mkdir")
@patch("scripts.pipeline.Path.exists")
def test_run_pipeline_success(
    mock_path_exists,
    mock_mkdir,
    mock_extract,
    mock_run_dbt,
    mock_read_mart,
    mock_train_model,
    mock_upload_file,
    mock_upload_json,
    mock_export_mart,
    mock_settings,
    sample_raw_df,
    sample_mart_df,
    mock_train_metrics,
):
    """Test du pipeline complet avec toutes les étapes mockées."""
    # Configuration des mocks
    mock_extract.return_value = sample_raw_df
    mock_read_mart.return_value = sample_mart_df

    # Simuler l'entraînement pour plusieurs modèles
    def train_side_effect(df, text_col, label_col, model_name, model_path):
        return mock_train_metrics(model_name)

    mock_train_model.side_effect = train_side_effect

    # Simuler l'existence des fichiers locaux pour l'upload
    mock_path_exists.return_value = True

    log = MagicMock()

    result = run_pipeline(log)

    # Vérifications des appels
    mock_extract.assert_called_once()
    mock_run_dbt.assert_called_once_with(log, target=DBT_TARGET)
    mock_read_mart.assert_called_once()

    # Vérifier que train_model a été appelé pour chaque modèle
    assert mock_train_model.call_count == len(MODELS_TO_TRAIN)

    # Vérifier les uploads (modèle best + second, rapport)
    # Au moins 1 appel pour le modèle best + 1 pour le second (top2)
    # + 2 pour models/model.joblib et models/model_<date>.joblib
    # + 2 pour les rapports JSON
    # + 1 pour l'export mart
    assert mock_upload_file.call_count >= 3
    assert mock_upload_json.call_count >= 1
    mock_export_mart.assert_called_once()

    # Vérifier le résultat
    assert result["raw_rows"] == 2
    assert result["mart_rows"] == 2
    assert result["best_model"] == "logreg"  # car logreg a 0.85 > 0.80
    assert result["accuracy"] == 0.85
    assert result["f1_macro"] == 0.84
    assert len(result["leaderboard"]) == len(MODELS_TO_TRAIN)

    # Vérifier que le log contient les messages clés
    log.assert_any_call("🚀 Démarrage pipeline SignalConso")
    log.assert_any_call("🏁 Pipeline terminé avec succès")
    log.assert_any_call("📊 Leaderboard des modèles :")


@patch("scripts.pipeline.train_model")
@patch("scripts.pipeline.read_mart_table")
@patch("scripts.pipeline._run_dbt")
@patch("scripts.pipeline.extract_from_signalconso_api")
def test_run_pipeline_no_model_success(
    mock_extract,
    mock_run_dbt,
    mock_read_mart,
    mock_train_model,
    mock_settings,
    sample_raw_df,
    sample_mart_df,
):
    """Lève RuntimeError si aucun modèle n'est entraîné avec succès."""
    mock_extract.return_value = sample_raw_df
    mock_read_mart.return_value = sample_mart_df
    mock_train_model.side_effect = Exception("Échec entraînement")

    log = MagicMock()

    with pytest.raises(RuntimeError, match="Aucun modèle entraîné avec succès."):
        run_pipeline(log)


@patch("scripts.pipeline.extract_from_signalconso_api")
def test_run_pipeline_extract_failure(mock_extract, mock_settings):
    """Propagation d'erreur si l'extraction échoue."""
    mock_extract.side_effect = Exception("API error")
    log = MagicMock()
    with pytest.raises(Exception, match="API error"):
        run_pipeline(log)


@patch("scripts.pipeline.read_mart_table")
@patch("scripts.pipeline._run_dbt")
@patch("scripts.pipeline.extract_from_signalconso_api")
def test_run_pipeline_dbt_failure(
    mock_extract,
    mock_run_dbt,
    mock_read_mart,
    mock_settings,
    sample_raw_df,
):
    """Propagation d'erreur si dbt run échoue."""
    mock_extract.return_value = sample_raw_df
    mock_run_dbt.side_effect = RuntimeError("dbt failed")
    log = MagicMock()
    with pytest.raises(RuntimeError, match="dbt failed"):
        run_pipeline(log)


@patch("scripts.pipeline.upload_file_to_gcs")
@patch("scripts.pipeline.train_model")
@patch("scripts.pipeline.read_mart_table")
@patch("scripts.pipeline._run_dbt")
@patch("scripts.pipeline.extract_from_signalconso_api")
def test_run_pipeline_upload_failure_does_not_stop_pipeline(
    mock_extract,
    mock_run_dbt,
    mock_read_mart,
    mock_train_model,
    mock_upload_file,
    mock_settings,
    sample_raw_df,
    sample_mart_df,
    mock_train_metrics,
):
    """Les échecs d'upload GCS sont loggés mais n'arrêtent pas le pipeline."""
    mock_extract.return_value = sample_raw_df
    mock_read_mart.return_value = sample_mart_df
    mock_train_model.side_effect = lambda **kwargs: mock_train_metrics(kwargs["model_name"])
    mock_upload_file.side_effect = Exception("GCS unavailable")

    log = MagicMock()
    result = run_pipeline(log)

    # Le pipeline doit réussir malgré les échecs d'upload
    assert result["best_model"] is not None
    # Vérifier que les erreurs ont été loggées
    assert any("⚠ Upload" in str(call) for call in log.call_args_list)

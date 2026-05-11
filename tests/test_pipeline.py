from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from scripts.pipeline import (
    DBT_EXECUTABLE,
    DBT_PROFILES_DIR,
    DBT_PROJECT_DIR,
    DBT_TARGET,
    MODELS_TO_TRAIN,
    _find_dbt,
    _print_leaderboard,
    _run_dbt,
    run_pipeline,
)


@pytest.fixture
def mock_settings(monkeypatch):
    settings = SimpleNamespace(
        GCS_BUCKET_NAME="test-bucket",
        GCP_PROJECT_ID="test-project",
    )
    monkeypatch.setattr("scripts.pipeline.get_settings", lambda: settings)
    return settings


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2],
            "text": ["bon produit", "service médiocre"],
            "category": ["alimentaire", "service"],
        }
    )


@pytest.fixture
def sample_mart_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clean_text": ["bon produit", "service médiocre"],
            "category": ["alimentaire", "service"],
            "is_valid": [True, True],
        }
    )


@pytest.fixture
def mock_train_metrics():
    def _metrics(model_name: str, best_model: str):
        score = 0.85 if model_name == best_model else 0.80
        f1 = 0.84 if model_name == best_model else 0.79
        return {
            "model_name": model_name,
            "accuracy": score,
            "f1_macro": f1,
            "n_train": 800,
            "n_test": 200,
            "n_classes": 3,
        }

    return _metrics


def test_find_dbt_venv(monkeypatch):
    monkeypatch.setattr("scripts.pipeline.shutil.which", lambda _: None)
    monkeypatch.setattr("scripts.pipeline.sys.executable", "/venv/bin/python")
    monkeypatch.setattr("scripts.pipeline.sys.platform", "linux")

    def fake_exists(self):
        return str(self).replace("\\", "/") == "/venv/bin/dbt"

    monkeypatch.setattr("scripts.pipeline.Path.exists", fake_exists)

    result = _find_dbt()
    assert str(result).replace("\\", "/") == "/venv/bin/dbt"


def test_find_dbt_system_path(monkeypatch):
    monkeypatch.setattr("scripts.pipeline.shutil.which", lambda _: "/usr/bin/dbt")
    monkeypatch.setattr("scripts.pipeline.sys.executable", "/venv/bin/python")
    monkeypatch.setattr("scripts.pipeline.sys.platform", "linux")

    def fake_exists(self):
        p = str(self).replace("\\", "/")
        return p == "/usr/bin/dbt"

    monkeypatch.setattr("scripts.pipeline.Path.exists", fake_exists)

    assert _find_dbt() == "/usr/bin/dbt"


def test_find_dbt_fallback_docker(monkeypatch):
    monkeypatch.setattr("scripts.pipeline.shutil.which", lambda _: None)
    monkeypatch.setattr("scripts.pipeline.sys.executable", "/venv/bin/python")
    monkeypatch.setattr("scripts.pipeline.sys.platform", "linux")

    def fake_exists(self):
        p = str(self).replace("\\", "/")
        return p in {"/usr/local/bin/dbt", "/usr/bin/dbt"}

    monkeypatch.setattr("scripts.pipeline.Path.exists", fake_exists)

    assert _find_dbt() == "/usr/local/bin/dbt"


def test_run_dbt_success(monkeypatch):
    mock_run = MagicMock()
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "Finished running"
    mock_run.return_value.stderr = ""

    monkeypatch.setattr("scripts.pipeline.subprocess.run", mock_run)
    monkeypatch.setattr(
        "scripts.pipeline.TemporaryDirectory",
        MagicMock(
            side_effect=[
                nullcontext("/tmp/dbt_logs"),
                nullcontext("/tmp/dbt_target"),
            ]
        ),
    )

    log = MagicMock()

    _run_dbt(log, target="dev")

    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    cmd = args[0]

    assert cmd[0] == DBT_EXECUTABLE
    assert cmd[1] == "run"
    assert "--target" in cmd
    assert cmd[cmd.index("--target") + 1] == "dev"
    assert kwargs["cwd"] == str(DBT_PROJECT_DIR.resolve())
    assert kwargs["env"]["DBT_PROFILES_DIR"] == DBT_PROFILES_DIR
    log.assert_any_call(f"  ▶ Exécutable : {DBT_EXECUTABLE}")
    log.assert_any_call(f"  ▶ Profiles   : {DBT_PROFILES_DIR}")
    log.assert_any_call("  ▶ Project    : " + str(DBT_PROJECT_DIR.resolve()))


def test_run_dbt_failure(monkeypatch):
    mock_run = MagicMock()
    mock_run.return_value.returncode = 1
    mock_run.return_value.stdout = "Error"
    mock_run.return_value.stderr = "Something wrong"

    monkeypatch.setattr("scripts.pipeline.subprocess.run", mock_run)
    monkeypatch.setattr(
        "scripts.pipeline.TemporaryDirectory",
        MagicMock(
            side_effect=[
                nullcontext("/tmp/dbt_logs"),
                nullcontext("/tmp/dbt_target"),
            ]
        ),
    )

    log = MagicMock()

    with pytest.raises(RuntimeError, match="dbt run a échoué"):
        _run_dbt(log)


def test_print_leaderboard(capsys):
    results = [
        {"model_name": "logreg", "accuracy": 0.85, "f1_macro": 0.84, "n_train": 800, "n_test": 200},
        {"model_name": "svc", "accuracy": 0.80, "f1_macro": 0.79, "n_train": 800, "n_test": 200},
    ]

    _print_leaderboard(results, print)
    captured = capsys.readouterr()

    assert "logreg" in captured.out
    assert "85.00%" in captured.out
    assert "svc" in captured.out
    assert "80.00%" in captured.out
    assert "🏆" in captured.out


def test_run_pipeline_success(
    monkeypatch,
    tmp_path,
    mock_settings,
    sample_raw_df,
    sample_mart_df,
    mock_train_metrics,
):
    fixed_today = datetime(2024, 1, 15)
    monkeypatch.setattr("scripts.pipeline.datetime", SimpleNamespace(utcnow=lambda: fixed_today))
    monkeypatch.chdir(tmp_path)

    mock_extract = MagicMock(return_value=sample_raw_df)
    mock_run_dbt = MagicMock()
    mock_read_mart = MagicMock(return_value=sample_mart_df)
    mock_upload_file = MagicMock()
    mock_upload_json = MagicMock()
    mock_export_mart = MagicMock(
        return_value="gs://test-bucket/processed/mart_signalconso_2024-01-15_*.csv"
    )

    monkeypatch.setattr("scripts.pipeline.extract_from_signalconso_api", mock_extract)
    monkeypatch.setattr("scripts.pipeline._run_dbt", mock_run_dbt)
    monkeypatch.setattr("scripts.pipeline.read_mart_table", mock_read_mart)
    monkeypatch.setattr("scripts.pipeline.upload_file_to_gcs", mock_upload_file)
    monkeypatch.setattr("scripts.pipeline.upload_json_to_gcs", mock_upload_json)
    monkeypatch.setattr("scripts.pipeline.export_mart_to_gcs", mock_export_mart)

    best_model = MODELS_TO_TRAIN[0]

    def train_side_effect(**kwargs):
        model_path = Path(kwargs["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("model", encoding="utf-8")
        return mock_train_metrics(kwargs["model_name"], best_model)

    monkeypatch.setattr("scripts.pipeline.train_model", MagicMock(side_effect=train_side_effect))

    log = MagicMock()
    result = run_pipeline(log)

    mock_extract.assert_called_once()
    mock_run_dbt.assert_called_once_with(log, target=DBT_TARGET)
    mock_read_mart.assert_called_once()
    assert result["raw_rows"] == 2
    assert result["mart_rows"] == 2
    assert result["best_model"] == best_model
    assert result["accuracy"] == 0.85
    assert result["f1_macro"] == 0.84
    assert len(result["leaderboard"]) == len(MODELS_TO_TRAIN)

    expected_upload_calls = 1 + min(2, len(MODELS_TO_TRAIN)) + 2
    assert mock_upload_file.call_count == expected_upload_calls
    assert mock_upload_json.call_count == 2
    mock_export_mart.assert_called_once()

    log.assert_any_call("🚀 Démarrage pipeline SignalConso")
    log.assert_any_call("🏁 Pipeline terminé avec succès")
    log.assert_any_call("\n📊 Leaderboard des modèles :")


def test_run_pipeline_no_model_success(
    monkeypatch,
    tmp_path,
    mock_settings,
    sample_raw_df,
    sample_mart_df,
):
    fixed_today = datetime(2024, 1, 15)
    monkeypatch.setattr("scripts.pipeline.datetime", SimpleNamespace(utcnow=lambda: fixed_today))
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        "scripts.pipeline.extract_from_signalconso_api", MagicMock(return_value=sample_raw_df)
    )
    monkeypatch.setattr("scripts.pipeline._run_dbt", MagicMock())
    monkeypatch.setattr("scripts.pipeline.read_mart_table", MagicMock(return_value=sample_mart_df))
    monkeypatch.setattr("scripts.pipeline.upload_file_to_gcs", MagicMock())
    monkeypatch.setattr("scripts.pipeline.upload_json_to_gcs", MagicMock())
    monkeypatch.setattr("scripts.pipeline.export_mart_to_gcs", MagicMock())
    monkeypatch.setattr(
        "scripts.pipeline.train_model", MagicMock(side_effect=Exception("Échec entraînement"))
    )

    log = MagicMock()

    with pytest.raises(RuntimeError, match="Aucun modèle entraîné avec succès\\."):
        run_pipeline(log)


def test_run_pipeline_extract_failure(monkeypatch, mock_settings):
    monkeypatch.setattr(
        "scripts.pipeline.extract_from_signalconso_api",
        MagicMock(side_effect=Exception("API error")),
    )

    log = MagicMock()

    with pytest.raises(Exception, match="API error"):
        run_pipeline(log)


def test_run_pipeline_dbt_failure(
    monkeypatch,
    tmp_path,
    mock_settings,
    sample_raw_df,
):
    fixed_today = datetime(2024, 1, 15)
    monkeypatch.setattr("scripts.pipeline.datetime", SimpleNamespace(utcnow=lambda: fixed_today))
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        "scripts.pipeline.extract_from_signalconso_api", MagicMock(return_value=sample_raw_df)
    )
    monkeypatch.setattr("scripts.pipeline.upload_file_to_gcs", MagicMock())
    monkeypatch.setattr(
        "scripts.pipeline._run_dbt", MagicMock(side_effect=RuntimeError("dbt failed"))
    )

    log = MagicMock()

    with pytest.raises(RuntimeError, match="dbt failed"):
        run_pipeline(log)


def test_run_pipeline_upload_failure_does_not_stop_pipeline(
    monkeypatch,
    tmp_path,
    mock_settings,
    sample_raw_df,
    sample_mart_df,
    mock_train_metrics,
):
    fixed_today = datetime(2024, 1, 15)
    monkeypatch.setattr("scripts.pipeline.datetime", SimpleNamespace(utcnow=lambda: fixed_today))
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        "scripts.pipeline.extract_from_signalconso_api", MagicMock(return_value=sample_raw_df)
    )
    monkeypatch.setattr("scripts.pipeline._run_dbt", MagicMock())
    monkeypatch.setattr("scripts.pipeline.read_mart_table", MagicMock(return_value=sample_mart_df))
    monkeypatch.setattr("scripts.pipeline.export_mart_to_gcs", MagicMock())

    best_model = MODELS_TO_TRAIN[0]

    def train_side_effect(**kwargs):
        model_path = Path(kwargs["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("model", encoding="utf-8")
        return mock_train_metrics(kwargs["model_name"], best_model)

    monkeypatch.setattr("scripts.pipeline.train_model", MagicMock(side_effect=train_side_effect))

    def upload_side_effect(bucket_name, local_path, blob_name):
        if blob_name.startswith("models/runs/"):
            raise Exception("GCS unavailable")
        return None

    monkeypatch.setattr(
        "scripts.pipeline.upload_file_to_gcs", MagicMock(side_effect=upload_side_effect)
    )
    monkeypatch.setattr("scripts.pipeline.upload_json_to_gcs", MagicMock())

    log = MagicMock()
    result = run_pipeline(log)

    assert result["best_model"] == best_model
    assert any("⚠ Upload" in str(call) for call in log.call_args_list)

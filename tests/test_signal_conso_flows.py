from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# On crée un mock pour simuler le bloc GCP
mock_block = MagicMock()
mock_block.get.return_value = "fake-secret-value"

with patch("prefect_gcp.credentials.GcpSecret.load", return_value=mock_block):
    from app.flows.signal_conso_flows import (
        _bool_series,
        _is_missing,
        _to_bool,
        apply_geo_filter_task,
        apply_temporal_filter_task,
        kpi_nombre_signalements_task,
        kpi_signalements_lus_reponse_task,
        kpi_signalements_transmis_lus_task,
        kpi_signalements_transmis_task,
        preprocess_task,
    )

# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

TODAY = date.today()


@pytest.fixture()
def df_base() -> pd.DataFrame:
    """DataFrame minimal avec les 3 colonnes booléennes et une date."""
    return pd.DataFrame(
        {
            "creationdate": [TODAY, TODAY, TODAY, TODAY],
            "signalement_transmis": [True, True, False, True],
            "signalement_lu": [True, False, False, True],
            "signalement_reponse": [True, False, False, False],
            "reg_name": ["Île-de-France", "Île-de-France", "PACA", "PACA"],
            "department_label": ["75 - Paris", "75 - Paris", "13 - BDR", "13 - BDR"],
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def test_is_missing_detects_none_empty_and_nan():
    assert _is_missing(None)
    assert _is_missing("")
    assert _is_missing("   ")
    assert _is_missing(float("nan"))
    assert not _is_missing(0)
    assert not _is_missing("ok")


def test_to_bool_handles_all_truthy_values():
    for v in ("1", "true", "oui", "yes", True, 1):
        assert _to_bool(v) is True


def test_to_bool_handles_falsy_and_missing():
    for v in ("0", "false", "non", None, "", False, 0):
        assert _to_bool(v) is False


def test_bool_series_converts_mixed_column():
    s = pd.Series(["oui", "non", "true", None, "1"])
    result = _bool_series(s)
    assert list(result) == [True, False, True, False, True]


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════


def test_preprocess_converts_creationdate(df_base):
    df_base["creationdate"] = df_base["creationdate"].astype(str)
    with patch("app.flows.signal_conso_flows.get_run_logger", return_value=MagicMock()):
        result = preprocess_task.fn(df_base)
    assert pd.api.types.is_datetime64_any_dtype(result["creationdate"])


def test_preprocess_converts_bool_columns(df_base):
    df_base["signalement_transmis"] = ["oui", "non", "oui", "1"]
    with patch("app.flows.signal_conso_flows.get_run_logger", return_value=MagicMock()):
        result = preprocess_task.fn(df_base)
    assert list(result["signalement_transmis"]) == [True, False, True, True]


def test_preprocess_creates_department_label_from_codes():
    df = pd.DataFrame({"dep_code": ["75", "13"], "dep_name": ["Paris", "BDR"]})
    with patch("app.flows.signal_conso_flows.get_run_logger", return_value=MagicMock()):
        result = preprocess_task.fn(df)
    assert "department_label" in result.columns
    assert result["department_label"].iloc[0] == "75 – Paris"


# ══════════════════════════════════════════════════════════════════════════════
# FILTRES
# ══════════════════════════════════════════════════════════════════════════════


def test_temporal_filter_mois_en_cours(df_base):
    df_base["creationdate"] = pd.to_datetime(df_base["creationdate"])
    with patch("app.flows.signal_conso_flows.get_run_logger", return_value=MagicMock()):
        result = apply_temporal_filter_task.fn(
            df_base, reference_date=TODAY, period="Depuis le début du mois"
        )
    assert not result.empty


def test_temporal_filter_exclut_dates_hors_periode(df_base):
    df_base["creationdate"] = pd.to_datetime(
        [TODAY - timedelta(days=30), TODAY - timedelta(days=20), TODAY, TODAY]
    )
    with patch("app.flows.signal_conso_flows.get_run_logger", return_value=MagicMock()):
        result = apply_temporal_filter_task.fn(
            df_base, reference_date=TODAY, period="7 derniers jours"
        )
    assert all(result["creationdate"].dt.date >= TODAY - timedelta(days=6))


def test_temporal_filter_toutes_donnees_ne_filtre_pas(df_base):
    df_base["creationdate"] = pd.to_datetime(df_base["creationdate"])
    with patch("app.flows.signal_conso_flows.get_run_logger", return_value=MagicMock()):
        result = apply_temporal_filter_task.fn(df_base, period="Toutes les données")
    assert len(result) == len(df_base)


def test_geo_filter_par_region(df_base):
    with patch("app.flows.signal_conso_flows.get_run_logger", return_value=MagicMock()):
        result = apply_geo_filter_task.fn(df_base, region="Île-de-France")
    assert all(result["reg_name"] == "Île-de-France")
    assert len(result) == 2


def test_geo_filter_par_departement(df_base):
    with patch("app.flows.signal_conso_flows.get_run_logger", return_value=MagicMock()):
        result = apply_geo_filter_task.fn(df_base, department_label="75 - Paris")
    assert len(result) == 2


# ══════════════════════════════════════════════════════════════════════════════
# KPIs
# ══════════════════════════════════════════════════════════════════════════════


def test_kpi_nombre_signalements(df_base):
    with patch("app.flows.signal_conso_flows.get_run_logger", return_value=MagicMock()):
        result = kpi_nombre_signalements_task.fn(df_base)
    assert result["kpi"] == "nombre_signalements"
    assert result["value"] == 4


def test_kpi_signalements_transmis(df_base):
    # 3 transmis sur 4 → 0.75
    with patch("app.flows.signal_conso_flows.get_run_logger", return_value=MagicMock()):
        result = kpi_signalements_transmis_task.fn(df_base)
    assert result["numerator"] == 3
    assert result["denominator"] == 4
    assert result["value"] == pytest.approx(0.75)


def test_kpi_signalements_transmis_colonne_manquante():
    df = pd.DataFrame({"autre": [1, 2]})
    with patch("app.flows.signal_conso_flows.get_run_logger", return_value=MagicMock()):
        result = kpi_signalements_transmis_task.fn(df)
    assert result["value"] is None
    assert "error" in result


def test_kpi_signalements_transmis_lus(df_base):
    # 3 transmis, 2 lus parmi les transmis → 2/3
    with patch("app.flows.signal_conso_flows.get_run_logger", return_value=MagicMock()):
        result = kpi_signalements_transmis_lus_task.fn(df_base)
    assert result["numerator"] == 2
    assert result["denominator"] == 3
    assert result["value"] == pytest.approx(2 / 3, rel=1e-4)


def test_kpi_signalements_lus_reponse(df_base):
    # 2 lus, 1 avec réponse → 0.5
    with patch("app.flows.signal_conso_flows.get_run_logger", return_value=MagicMock()):
        result = kpi_signalements_lus_reponse_task.fn(df_base)
    assert result["numerator"] == 1
    assert result["denominator"] == 2
    assert result["value"] == pytest.approx(0.5)


def test_kpi_lus_reponse_zero_division():
    df = pd.DataFrame({"signalement_lu": [False, False], "signalement_reponse": [False, False]})
    with patch("app.flows.signal_conso_flows.get_run_logger", return_value=MagicMock()):
        result = kpi_signalements_lus_reponse_task.fn(df)
    assert result["value"] == 0.0

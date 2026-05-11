from __future__ import annotations

import shutil
import sys
from pathlib import Path

from app.ml.train import AVAILABLE_MODELS

API_URL = "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/signalconso/records"
GCS_RAW_PREFIX = "raw/"

ROOT_DIR = Path(__file__).resolve().parents[1]
DBT_PROJECT_DIR = ROOT_DIR / "dbt"
DBT_TARGET = "dev"

# profiles.yml : priorité au dossier dbt/ du projet (Docker),
# fallback sur ~/.dbt (Windows local)
_dbt_profiles_in_project = DBT_PROJECT_DIR / "profiles.yml"
DBT_PROFILES_DIR = (
    str(DBT_PROJECT_DIR) if _dbt_profiles_in_project.exists() else str(Path.home() / ".dbt")
)


# Exécutable dbt : venv Windows → système PATH → /usr/local/bin (Docker)
def _find_dbt() -> str:
    # 1. même dossier que le python courant (venv Windows/Linux)
    venv_dbt = Path(sys.executable).parent / ("dbt.exe" if sys.platform == "win32" else "dbt")
    if venv_dbt.exists():
        return str(venv_dbt)
    # 2. PATH système (Docker, CI/CD)
    which = shutil.which("dbt")
    if which:
        return which
    # 3. fallback absolu Docker
    for p in ["/usr/local/bin/dbt", "/usr/bin/dbt"]:
        if Path(p).exists():
            return p
    return "dbt"  # laisse le shell se débrouiller


DBT_EXECUTABLE = _find_dbt()
MODELS_TO_TRAIN = list(AVAILABLE_MODELS.keys())

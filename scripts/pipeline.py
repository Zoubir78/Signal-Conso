from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

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


# ─────────────────────────────────────────────────────────────────────────────
# DBT RUNNER
# ─────────────────────────────────────────────────────────────────────────────


def _run_dbt(log, target: str = DBT_TARGET) -> None:
    project_dir = str(DBT_PROJECT_DIR.resolve())

    env = os.environ.copy()
    env["DBT_PROFILES_DIR"] = DBT_PROFILES_DIR

    gac = env.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if gac:
        env["GOOGLE_APPLICATION_CREDENTIALS"] = str(Path(gac).resolve())

    with (
        TemporaryDirectory(prefix="dbt_logs_") as log_dir,
        TemporaryDirectory(prefix="dbt_target_") as target_dir,
    ):
        cmd = [
            DBT_EXECUTABLE,
            "run",
            "--log-path",
            log_dir,
            "--target-path",
            target_dir,
            "--profiles-dir",
            DBT_PROFILES_DIR,
            "--target",
            target,
        ]

        log(f"  ▶ Exécutable : {DBT_EXECUTABLE}")
        log(f"  ▶ Profiles   : {DBT_PROFILES_DIR}")
        log(f"  ▶ Project    : {project_dir}")
        log(f"  ▶ Log path   : {log_dir}")
        log(f"  ▶ Target path: {target_dir}")
        log(f"  ▶ Commande   : {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=project_dir,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if result.stdout.strip():
            log("----- dbt stdout -----")
            log(result.stdout.strip()[-3000:])
        if result.stderr.strip():
            log("----- dbt stderr -----")
            log(result.stderr.strip()[-1500:])

        if result.returncode != 0:
            raise RuntimeError("dbt run a échoué (voir les logs stdout/stderr ci-dessus)")


# ─────────────────────────────────────────────────────────────────────────────
# LEADERBOARD
# ─────────────────────────────────────────────────────────────────────────────


def _print_leaderboard(results: list[dict], log) -> None:
    log("\n  ┌─────────────────────────┬──────────┬──────────┬─────────┬────────┐")
    log("    │ Modèle                  │ Accuracy │ F1-macro │  Train  │  Test  │")
    log("    ├─────────────────────────┼──────────┼──────────┼─────────┼────────┤")
    for i, r in enumerate(sorted(results, key=lambda x: x["accuracy"], reverse=True), 1):
        crown = "🏆" if i == 1 else "   "
        log(
            f"  │ {crown}{r['model_name']:<21} │"
            f"  {r['accuracy']:.2%}  │"
            f"  {r.get('f1_macro', 0):.2%}  │"
            f" {r['n_train']:>6}  │"
            f" {r['n_test']:>5}  │"
        )
    log("    └─────────────────────────┴──────────┴──────────┴─────────┴────────┘\n")

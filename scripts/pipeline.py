from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from app.core.config import get_settings
from app.ingestion.extract import extract_from_signalconso_api
from app.ml.train import AVAILABLE_MODELS, train_model
from app.services.bigquery_service import read_mart_table
from app.services.gcs_service import upload_file_to_gcs

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


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────


def run_pipeline(log) -> dict:
    """
    Pipeline complet SignalConso — appelé depuis Streamlit ou CLI.

    Flux :
      ① Extract API (10 000 enregistrements)
      ② Upload GCS raw/  →  table externe BigQuery lit automatiquement
      ③ dbt run  →  staging → intermediate → mart_signalconso
      ④ Lecture mart BigQuery
      ⑤ Entraînement multi-modèles (TF-IDF + LogReg · SGD · SVC · NB · RF)
      ⑥ Leaderboard + sélection du meilleur
      ⑦ Upload GCS models/ + rapport JSON

    Returns:
        dict : raw_rows, mart_rows, best_model, accuracy, f1_macro,
               n_classes, leaderboard, date
    """
    settings = get_settings()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    log("🚀 Démarrage pipeline SignalConso")

    # ── ① EXTRACT ─────────────────────────────────────────────────────────────
    log("📥 Extraction API SignalConso...")
    raw_df = extract_from_signalconso_api(API_URL, limit=10_000)
    log(f"  ✔ {len(raw_df):,} enregistrements extraits")

    # ── ② UPLOAD GCS ──────────────────────────────────────────────────────────
    raw_path = Path("data/raw/signalconso.csv")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_path, index=False)

    gcs_blob = f"{GCS_RAW_PREFIX}signalconso_{today}.csv"
    upload_file_to_gcs(settings.GCS_BUCKET_NAME, str(raw_path), gcs_blob)
    log(f"  ✔ RAW uploadé → gs://{settings.GCS_BUCKET_NAME}/{gcs_blob}")
    log("  ℹ Table externe BigQuery lit ce fichier automatiquement")

    # ── ③ DBT ─────────────────────────────────────────────────────────────────
    log("🔧 Modélisation dbt (staging → intermediate → mart)...")
    _run_dbt(log, target=DBT_TARGET)
    log("  ✔ dbt run terminé")

    # ── ④ LECTURE DU MART ─────────────────────────────────────────────────────
    log("☁️  Lecture du mart dbt depuis BigQuery...")
    mart_df = read_mart_table(
        project_id=settings.GCP_PROJECT_ID,
        filters="is_valid = TRUE AND category IS NOT NULL",
    )
    log(f"  ✔ {len(mart_df):,} lignes prêtes pour l'entraînement")

    mart_path = Path("data/processed/signalconso_mart.csv")
    mart_path.parent.mkdir(parents=True, exist_ok=True)
    mart_df.to_csv(mart_path, index=False)

    # ── ⑤ ENTRAÎNEMENT MULTI-MODÈLES ──────────────────────────────────────────
    log(f"🤖 Entraînement de {len(MODELS_TO_TRAIN)} modèles...")
    all_results: list[dict] = []

    for model_name in MODELS_TO_TRAIN:
        log(f"  ▶ {model_name}…")
        model_path = models_dir / f"{model_name}.joblib"
        try:
            metrics = train_model(
                df=mart_df,
                text_col="clean_text",
                label_col="category",
                model_name=model_name,
                model_path=str(model_path),
            )
            all_results.append(metrics)
            log(
                f"    ✔ Accuracy {metrics['accuracy']:.2%} · "
                f"F1-macro {metrics.get('f1_macro', 0):.2%} · "
                f"{metrics['n_train']:,} train / {metrics['n_test']:,} test"
            )
        except Exception as e:
            log(f"    ✖ {model_name} échoué : {e}")

    if not all_results:
        raise RuntimeError("Aucun modèle entraîné avec succès.")

    # ── ⑥ LEADERBOARD & SÉLECTION ─────────────────────────────────────────────
    log("\n📊 Leaderboard des modèles :")
    all_results.sort(key=lambda r: r["accuracy"], reverse=True)
    _print_leaderboard(all_results, log)

    best = all_results[0]

    log(
        f"🏆 Meilleur modèle : {best['model_name']} "
        f"(accuracy={best['accuracy']:.2%} · f1-macro={best.get('f1_macro', 0):.2%})"
    )

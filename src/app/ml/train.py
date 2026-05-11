from __future__ import annotations

from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ── Catalogue des modèles disponibles ────────────────────────────────────────
# Chaque entrée est un callable qui retourne un estimator sklearn non entraîné.
# TfidfVectorizer est partagé et ajouté automatiquement dans build_pipeline().

AVAILABLE_MODELS: dict[str, Any] = {
    "logreg": LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="auto",
        n_jobs=-1,
    ),
    "sgd": SGDClassifier(
        loss="modified_huber",  # supporte predict_proba
        max_iter=200,
        tol=1e-3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
    "linearsvc": CalibratedClassifierCV(
        LinearSVC(
            C=0.5,
            max_iter=2000,
            class_weight="balanced",
        ),
        cv=3,
    ),
    "complementnb": ComplementNB(alpha=0.1),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
}


def _tfidf() -> TfidfVectorizer:
    """Retourne un TfidfVectorizer partagé par tous les pipelines."""
    return TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )


def build_pipeline(model_name: str) -> Pipeline:
    """
    Construit le pipeline sklearn : TF-IDF → classificateur.

    Args:
        model_name: Clé dans AVAILABLE_MODELS.

    Returns:
        Pipeline sklearn non entraîné.

    Raises:
        ValueError: Si model_name est inconnu.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Modèle inconnu : '{model_name}'. Disponibles : {list(AVAILABLE_MODELS)}")
    return Pipeline(
        [
            ("tfidf", _tfidf()),
            ("clf", AVAILABLE_MODELS[model_name]),
        ]
    )

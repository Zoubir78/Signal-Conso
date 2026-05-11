from __future__ import annotations

from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
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

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter()

settings = get_settings()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

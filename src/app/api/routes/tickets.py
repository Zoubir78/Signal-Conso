from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings
from app.ml.predict import TicketModel

router = APIRouter()

settings = get_settings()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

# Le best model est celui uploadé par le pipeline dans models/model.joblib
model = TicketModel(
    model_path=str(MODEL_PATH),
    bucket_name=settings.GCS_BUCKET_NAME,
    blob_name="models/model.joblib",
)


class TicketRequest(BaseModel):
    text: str


class TicketResponse(BaseModel):
    predicted_category: str
    confidence: float

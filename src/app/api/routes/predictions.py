from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.api.schemas.prediction import PredictionRequest, PredictionResponse
from app.core.config import get_settings
from app.ml.predict import TicketModel, normalize_text
from app.services.gcs_service import find_prediction_in_bucket, upload_json_to_gcs

router = APIRouter()

settings = get_settings()
model = TicketModel(settings.MODEL_PATH)
MODEL_VERSION = settings.MODEL_VERSION


def get_model() -> TicketModel:
    if model.model is None:
        model.load()
    return model


# --------- CREATE PREDICTION ---------
@router.post("", response_model=PredictionResponse)
def create_prediction(request: PredictionRequest):
    try:
        prediction_id = str(uuid4())

        clean_text = normalize_text(request.text)
        loaded_model = get_model()
        prediction, confidence = loaded_model.predict_with_proba(request.text)

        now = datetime.utcnow()
        blob_path = f"predictions/{now.year}/{now.month:02d}/{now.day:02d}/{prediction_id}.json"

        data = {
            "id": prediction_id,
            "input_text": request.text,
            "clean_text": clean_text,
            "predicted_category": prediction,
            "confidence": float(confidence),
            "model_version": MODEL_VERSION,
            "created_at": now.isoformat(),
        }

        upload_json_to_gcs(
            bucket_name=settings.GCS_BUCKET_NAME,
            blob_name=blob_path,
            data=data,
        )

        return PredictionResponse(**data)

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# --------- GET PREDICTION ---------
@router.get("/{prediction_id}", response_model=PredictionResponse)
def get_prediction(prediction_id: str):
    data = find_prediction_in_bucket(
        bucket_name=settings.GCS_BUCKET_NAME,
        prediction_id=prediction_id,
    )

    if data is None:
        raise HTTPException(status_code=404, detail="Prédiction introuvable")

    return PredictionResponse(**data)

from __future__ import annotations

import os

from dotenv import load_dotenv

# charge .env automatiquement
load_dotenv()


class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "SignalConso")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/model.joblib")
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "logreg-v1")

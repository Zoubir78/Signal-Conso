from __future__ import annotations

import os

from dotenv import load_dotenv

# charge .env automatiquement
load_dotenv()


class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "SignalConso App")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

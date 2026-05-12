from __future__ import annotations

from pathlib import Path

import joblib
from google.cloud import storage


class TicketModel:
    def __init__(
        self, model_path: str, bucket_name: str | None = None, blob_name: str | None = None
    ):
        self.model_path = model_path
        self.bucket_name = bucket_name
        self.blob_name = blob_name
        self.model = None

    def download_from_gcs(self) -> None:
        if not self.bucket_name or not self.blob_name:
            raise FileNotFoundError(f"Modèle introuvable localement : {self.model_path}")

        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(self.blob_name)

        if not blob.exists():
            raise FileNotFoundError(
                f"Modèle introuvable dans GCS : gs://{self.bucket_name}/{self.blob_name}"
            )

        path = Path(self.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(path))

    def load(self) -> None:
        path = Path(self.model_path)

        if not path.exists():
            self.download_from_gcs()

        if not path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {self.model_path}")

        self.model = joblib.load(path)

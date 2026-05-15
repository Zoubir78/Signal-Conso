from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import requests


def extract_from_signalconso_api(
    api_url: str,
    limit: int = 100_000,
    date_from: date | None = None,  # ← nouveau paramètre
) -> pd.DataFrame:
    page_size = 100
    offset = 0
    rows = []

    # Filtre sur 2 ans de données si pas de date précisée
    if date_from is None:
        date_from = date.today() - timedelta(days=730)

    while len(rows) < limit:
        params = {
            "limit": page_size,
            "offset": offset,
            "order_by": "-dateCreation",
            "where": f"dateCreation >= '{date_from.isoformat()}'",  # ← filtre API
        }
        response = requests.get(api_url, params=params, timeout=60)
        response.raise_for_status()
        records = response.json().get("results", [])
        if not records:
            break
        rows.extend(records)
        offset += page_size

    return pd.json_normalize(rows[:limit])

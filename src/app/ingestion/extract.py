from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import requests


def extract_from_signalconso_api(
    api_url: str,
    limit: int = 50_000,
    months_back: int = 12,  # ← fenêtre glissante par défaut
) -> pd.DataFrame:
    """
    Pagination par tranches de dates pour contourner la limite offset=10 000
    de l'API OpenDataSoft.
    """
    page_size = 100
    max_offset = 9_900  # ← limite dure ODS
    rows: list[dict] = []

    end_dt = date.today()
    start_dt = end_dt - timedelta(days=30 * months_back)

    # Découper en tranches de 7 jours
    cursor = start_dt
    while cursor < end_dt and len(rows) < limit:
        next_cursor = min(cursor + timedelta(days=7), end_dt)

        offset = 0
        while len(rows) < limit and offset <= max_offset:
            params = {
                "limit": page_size,
                "offset": offset,
                "order_by": "-creationdate",
                "where": (
                    f"creationdate >= '{cursor.isoformat()}'"
                    f" AND creationdate < '{next_cursor.isoformat()}'"
                ),
            }
            response = requests.get(api_url, params=params, timeout=60)
            response.raise_for_status()

            records = response.json().get("results", [])
            if not records:
                break

            rows.extend(records)
            offset += page_size

        cursor = next_cursor

    if not rows:
        return pd.DataFrame()

    return pd.json_normalize(rows[:limit])

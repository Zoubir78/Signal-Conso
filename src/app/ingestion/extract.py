from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import requests


def extract_from_signalconso_api(
    api_url: str,
    limit: int = 100_000,
    date_from: date | None = None,
) -> pd.DataFrame:
    """
    Stratégie : on pagine par tranches de dates (1 mois) au lieu d'utiliser offset.
    """
    page_size = 100
    rows: list[dict] = []

    # Plage de dates : 2 ans par défaut
    end_dt = datetime.now(UTC)
    start_dt = (
        datetime(date_from.year, date_from.month, date_from.day, tzinfo=UTC)
        if date_from
        else end_dt - timedelta(days=730)
    )

    # Découper en tranches mensuelles
    cursor = start_dt
    while cursor < end_dt and len(rows) < limit:
        next_cursor = min(cursor + timedelta(days=30), end_dt)

        date_start_str = cursor.strftime("%Y-%m-%dT%H:%M:%SZ")
        date_end_str = next_cursor.strftime("%Y-%m-%dT%H:%M:%SZ")

        offset = 0
        while len(rows) < limit:
            params = {
                "limit": page_size,
                "offset": offset,
                "where": (
                    f"creationdate >= '{date_start_str}' AND creationdate < '{date_end_str}'"
                ),
            }
            response = requests.get(api_url, params=params, timeout=60)
            response.raise_for_status()

            payload = response.json()
            records = payload.get("results", [])
            if not records:
                break

            rows.extend(records)
            offset += page_size

            # Sécurité : offset max ODS = 9 900
            if offset >= 9_900:
                break

        cursor = next_cursor

    if not isinstance(rows, list):
        raise ValueError("Format inattendu retourné par l'API SignalConso")

    return pd.json_normalize(rows[:limit])

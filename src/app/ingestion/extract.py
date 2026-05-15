from __future__ import annotations

from datetime import date

import pandas as pd
import requests


def extract_from_signalconso_api(
    api_url: str,
    limit: int = 10_000,
    date_from: date | None = None,
    date_to: date | None = None,
) -> pd.DataFrame:
    page_size = 100
    offset = 0
    rows = []

    while len(rows) < limit:
        params = {
            "limit": page_size,
            "offset": offset,
            "order_by": "-creationdate",
        }
        response = requests.get(api_url, params=params, timeout=60)
        response.raise_for_status()
        records = response.json().get("results", [])
        if not records:
            break
        rows.extend(records)
        offset += page_size

    df = pd.json_normalize(rows)

    if df.empty:
        return df

    date_col = next(
        (c for c in df.columns if "date" in c.lower() and "creation" in c.lower()),
        None,
    )

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        if date_from:
            df = df[df[date_col].dt.date >= date_from]

        if date_to:
            df = df[df[date_col].dt.date <= date_to]

        df = df.sort_values(date_col, ascending=False).head(limit).reset_index(drop=True)
    else:
        df = df.head(limit).reset_index(drop=True)

    return df

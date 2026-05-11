from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CleaningConfig:
    min_text_length: int = 10
    include_category_in_text: bool = False  # laisser False pour éviter la fuite de cible


COLUMN_ALIASES = {
    "source_id": ["source_id", "id", "recordid", "record_id", "uuid"],
    "creationdate": [
        "creationdate",
        "created_at",
        "date",
        "created",
        "publication_date",
    ],
    "category": ["category", "categorie", "catégorie", "theme", "main_theme"],
    "subcategories": [
        "subcategories",
        "sub_category",
        "subcategory",
        "sub_theme",
        "sous_categorie",
        "sous_theme",
    ],
    "tags": ["tags", "tag"],
    "status": ["status", "statut", "state"],
    "dep_name": ["dep_name", "department_name", "departement_name", "department"],
    "dep_code": ["dep_code", "department_code"],
    "reg_name": ["reg_name", "region_name", "région"],
    "reg_code": ["reg_code", "region_code"],
    "complaint_text": [
        "complaint_text",
        "description",
        "narrative",
        "message",
        "content",
        "body",
        "details",
    ],
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

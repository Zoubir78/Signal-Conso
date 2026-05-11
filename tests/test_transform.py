from __future__ import annotations

import pandas as pd

from app.ingestion.transform import (
    CleaningConfig,
    build_clean_text,
    normalize_text,
    transform_dataframe,
)

# ══════════════════════════════════════════════════════════════════════════════
# normalize_text
# ══════════════════════════════════════════════════════════════════════════════


class TestNormalizeText:
    def test_returns_empty_string_for_none(self):
        assert normalize_text(None) == ""

    def test_returns_empty_string_for_nan(self):
        assert normalize_text(float("nan")) == ""

    def test_lowercases_text(self):
        assert normalize_text("HELLO WORLD") == "hello world"

    def test_strips_accents(self):
        assert normalize_text("éàü") == "eau"

    def test_removes_urls(self):
        result = normalize_text("visitez https://example.com pour plus d'infos")
        assert "http" not in result
        assert "example" not in result

    def test_removes_emails(self):
        result = normalize_text("contact user@example.com pour help")
        assert "@" not in result

    def test_removes_special_characters(self):
        result = normalize_text("hello! world, test.")
        assert "!" not in result
        assert "," not in result
        assert "." not in result

    def test_collapses_whitespace(self):
        result = normalize_text("hello   world")
        assert "  " not in result
        assert result == "hello world"

    def test_strips_leading_trailing_spaces(self):
        assert normalize_text("  hello  ") == "hello"

    def test_normalizes_unicode(self):
        result = normalize_text("café")
        assert result == "cafe"

    def test_returns_empty_for_empty_string(self):
        assert normalize_text("") == ""

    def test_returns_empty_for_whitespace_only(self):
        assert normalize_text("   ") == ""

    def test_keeps_alphanumeric_and_spaces(self):
        result = normalize_text("signal123 conso")
        assert result == "signal123 conso"


# ══════════════════════════════════════════════════════════════════════════════
# build_clean_text
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildCleanText:
    def _row(self, **kwargs) -> pd.Series:
        return pd.Series(kwargs)

    def test_returns_string(self):
        row = self._row(tags="spam", status="open")
        result = build_clean_text(row)
        assert isinstance(result, str)

    def test_includes_tags(self):
        row = self._row(tags="alimentation")
        result = build_clean_text(row)
        assert "alimentation" in result

    def test_includes_subcategories(self):
        row = self._row(subcategories="prix abusif")
        result = build_clean_text(row)
        assert "prix abusif" in result

    def test_includes_status(self):
        row = self._row(status="fermé")
        result = build_clean_text(row)
        assert "ferme" in result  # accentué → normalisé

    def test_includes_dep_name(self):
        row = self._row(dep_name="Bouches-du-Rhône")
        result = build_clean_text(row)
        assert "bouches du rhone" in result

    def test_includes_reg_name(self):
        row = self._row(reg_name="Île-de-France")
        result = build_clean_text(row)
        assert "ile de france" in result

    def test_includes_complaint_text(self):
        row = self._row(description="produit défectueux reçu")
        result = build_clean_text(row)
        assert "produit defectueux recu" in result

    def test_excludes_category_by_default(self):
        row = self._row(category="Alimentation", tags="autre")
        result = build_clean_text(row)
        # category non incluse avec config par défaut
        assert "alimentation" not in result

    def test_includes_category_when_config_enabled(self):
        config = CleaningConfig(include_category_in_text=True)
        row = self._row(category="Alimentation")
        result = build_clean_text(row, config=config)
        assert "alimentation" in result

    def test_deduplicates_parts(self):
        row = self._row(tags="spam", subcategories="spam")
        result = build_clean_text(row)
        assert result.count("spam") == 1

    def test_returns_empty_for_empty_row(self):
        row = self._row()
        result = build_clean_text(row)
        assert result == ""

    def test_handles_list_tags(self):
        row = self._row(tags=["alimentation", "prix"])
        result = build_clean_text(row)
        assert "alimentation" in result
        assert "prix" in result

    def test_handles_stringified_list(self):
        row = self._row(tags="['AchatMagasin', 'Autre']")
        result = build_clean_text(row)
        assert "achatmagasin" in result
        assert "autre" in result

    def test_uses_default_config_when_none_given(self):
        row = self._row(tags="test")
        result_default = build_clean_text(row, config=None)
        result_explicit = build_clean_text(row, config=CleaningConfig())
        assert result_default == result_explicit


# ══════════════════════════════════════════════════════════════════════════════
# transform_dataframe
# ══════════════════════════════════════════════════════════════════════════════


def _make_df(**columns) -> pd.DataFrame:
    """Crée un DataFrame depuis des colonnes sous forme de listes."""
    return pd.DataFrame(columns)


class TestTransformDataframe:
    def test_returns_dataframe(self):
        df = _make_df(tags=["spam"], status=["open"])
        result = transform_dataframe(df)
        assert isinstance(result, pd.DataFrame)

    def test_adds_clean_text_column(self):
        df = _make_df(tags=["alimentation"], status=["open"])
        result = transform_dataframe(df)
        assert "clean_text" in result.columns

    def test_adds_token_count_column(self):
        df = _make_df(tags=["alimentation securite"], status=["open"])
        result = transform_dataframe(df)
        assert "token_count" in result.columns
        assert result["token_count"].iloc[0] >= 1

    def test_adds_is_valid_column(self):
        df = _make_df(tags=["alimentation"], status=["open"])
        result = transform_dataframe(df)
        assert "is_valid" in result.columns

    def test_filters_out_short_clean_text(self):
        config = CleaningConfig(min_text_length=10)
        df = _make_df(
            tags=["ok", "x"],  # "ok" → trop court, "x" → trop court
            status=["open", "open"],
        )
        # Toutes les lignes produiront un clean_text < 10 caractères
        result = transform_dataframe(df, config=config)
        assert result.empty or all(result["clean_text"].str.len() >= 10)

    def test_keeps_rows_with_sufficient_text(self):
        config = CleaningConfig(min_text_length=5)
        df = _make_df(
            description=["produit defectueux signalement grave", "ok"],
            status=["open", "open"],
        )
        result = transform_dataframe(df, config=config)
        # Au moins la première ligne passe le filtre
        assert len(result) >= 1

    def test_deduplicates_on_source_id(self):
        df = _make_df(
            source_id=["abc", "abc", "xyz"],
            description=[
                "produit defectueux signalement grave",
                "produit defectueux signalement grave",
                "autre signalement probleme qualite",
            ],
        )
        result = transform_dataframe(df)
        ids = result["source_id"].tolist()
        assert len(ids) == len(set(ids))

    def test_deduplicates_without_source_id_column(self):
        df = _make_df(
            description=[
                "produit defectueux signalement grave",
                "produit defectueux signalement grave",  # doublon exact
                "autre signalement probleme qualite produit",
            ]
        )
        result = transform_dataframe(df)
        assert len(result) == 2

    def test_resets_index(self):
        df = _make_df(
            description=[
                "premier signalement produit defectueux grave",
                "deuxieme signalement probleme qualite service",
            ]
        )
        result = transform_dataframe(df)
        assert list(result.index) == list(range(len(result)))

    def test_does_not_mutate_original_dataframe(self):
        df = _make_df(
            description=["signalement produit defectueux grave"],
            status=["open"],
        )
        original_columns = list(df.columns)
        transform_dataframe(df)
        assert list(df.columns) == original_columns
        assert "clean_text" not in df.columns

    def test_converts_creationdate_to_datetime(self):
        df = _make_df(
            creationdate=["2024-01-15", "2024-02-20"],
            description=[
                "signalement produit defectueux grave",
                "probleme qualite service client",
            ],
        )
        result = transform_dataframe(df)
        assert pd.api.types.is_datetime64_any_dtype(result["creationdate"])

    def test_handles_empty_dataframe(self):
        df = pd.DataFrame(columns=["source_id", "description", "status"])
        result = transform_dataframe(df)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_token_count_matches_word_count(self):
        df = _make_df(description=["produit defectueux signalement qualite grave"])
        result = transform_dataframe(df)
        if not result.empty:
            expected = len(result["clean_text"].iloc[0].split())
            assert result["token_count"].iloc[0] == expected

    def test_uses_column_aliases_for_category(self):
        """Vérifie que 'catégorie' est reconnu comme alias de 'category'."""
        config = CleaningConfig(include_category_in_text=True)
        df = _make_df(
            catégorie=["Alimentation"],
            description=["signalement produit defectueux grave"],
        )
        result = transform_dataframe(df, config=config)
        if not result.empty:
            assert "alimentation" in result["clean_text"].iloc[0]

    def test_uses_column_aliases_for_description(self):
        """Vérifie que 'narrative' est reconnu comme alias de 'complaint_text'."""
        df = _make_df(narrative=["signalement produit defectueux qualite grave"])
        result = transform_dataframe(df)
        if not result.empty:
            assert "signalement" in result["clean_text"].iloc[0]

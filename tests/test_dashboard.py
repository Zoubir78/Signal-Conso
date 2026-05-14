from pathlib import Path

DASHBOARD_PATH = Path("src/app/dashboard/dashboard.py")


def test_realtime_tab_is_inserted_next_to_overview():
    text = DASHBOARD_PATH.read_text(encoding="utf-8")

    assert "Flows temps réel" in text
    assert "Vue d'ensemble" in text
    assert text.index("Vue d'ensemble") < text.index("Flows temps réel")

    # The Prefect controls should no longer live in the sidebar.
    assert "Prefect Orchestration" not in text
    assert "Déploiement à lancer" not in text

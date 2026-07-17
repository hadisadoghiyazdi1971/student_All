from models import AppSettings
from settings_service import SettingsStore


def test_settings_are_persisted(tmp_path):
    store = SettingsStore(tmp_path / "settings.db")
    expected = AppSettings(top_articles=55, top_people=7, keyword_count=9)
    store.save(expected, "user-a")
    assert store.load("user-a") == expected
    assert store.load("unknown") == AppSettings()

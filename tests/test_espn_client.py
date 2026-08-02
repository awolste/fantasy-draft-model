import pytest

from ffdraft.sources.espn import EspnClient, MissingCredentials


def test_missing_credentials_raise_a_clear_error(monkeypatch):
    monkeypatch.delenv("ESPN_S2", raising=False)
    monkeypatch.delenv("ESPN_SWID", raising=False)
    monkeypatch.delenv("FF_LEAGUE_ID", raising=False)
    monkeypatch.setattr("ffdraft.sources.espn._load_dotenv", lambda: None)
    with pytest.raises(MissingCredentials, match="ESPN_S2"):
        EspnClient.from_env()


def test_historical_url_uses_league_history_endpoint():
    client = EspnClient(league_id=123, espn_s2="s2", swid="{sw}")
    url, params = client.season_request(2019, views=["mDraftDetail"])
    assert "leagueHistory/123" in url
    assert params["seasonId"] == 2019
    assert params["view"] == ["mDraftDetail"]


def test_current_season_url_uses_seasons_endpoint():
    client = EspnClient(league_id=123, espn_s2="s2", swid="{sw}")
    url, params = client.season_request(2026, views=["mTeam"], current_season=2026)
    assert "seasons/2026/segments/0/leagues/123" in url
    assert "seasonId" not in params


def test_cookies_include_both_required_values():
    client = EspnClient(league_id=123, espn_s2="s2", swid="{sw}")
    assert client.cookies == {"espn_s2": "s2", "SWID": "{sw}"}

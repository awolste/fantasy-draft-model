"""Authenticated ESPN Fantasy v3 client.

Returns raw JSON only -- parsing lives in espn_parse.py so that saved JSON
fixtures can be reparsed without network access.

ESPN exposes completed seasons under a different path than the active one:
  completed: /leagueHistory/{league_id}?seasonId={year}
  active:    /seasons/{year}/segments/0/leagues/{league_id}
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import requests

BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "espn"
ENV_PATH = Path(__file__).resolve().parents[3] / ".env"


class MissingCredentials(RuntimeError):
    pass


def _load_dotenv() -> None:
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class EspnClient:
    league_id: int
    espn_s2: str
    swid: str

    @classmethod
    def from_env(cls) -> "EspnClient":
        _load_dotenv()
        missing = [k for k in ("FF_LEAGUE_ID", "ESPN_S2", "ESPN_SWID") if not os.environ.get(k)]
        if missing:
            raise MissingCredentials(
                f"Missing environment variables: {', '.join(missing)}. "
                f"Copy .env.example to .env and fill in your ESPN cookies."
            )
        return cls(
            league_id=int(os.environ["FF_LEAGUE_ID"]),
            espn_s2=os.environ["ESPN_S2"],
            swid=os.environ["ESPN_SWID"],
        )

    @property
    def cookies(self) -> dict[str, str]:
        return {"espn_s2": self.espn_s2, "SWID": self.swid}

    def season_request(
        self, season: int, views: list[str], current_season: int | None = None
    ) -> tuple[str, dict]:
        if current_season is not None and season == current_season:
            return f"{BASE}/seasons/{season}/segments/0/leagues/{self.league_id}", {"view": views}
        return f"{BASE}/leagueHistory/{self.league_id}", {"seasonId": season, "view": views}

    def fetch_season(self, season: int, views: list[str], current_season: int | None = None) -> dict:
        url, params = self.season_request(season, views, current_season)
        response = requests.get(url, params=params, cookies=self.cookies, timeout=30)
        response.raise_for_status()
        payload = response.json()
        # leagueHistory returns a single-element list; the active season returns an object.
        return payload[0] if isinstance(payload, list) else payload

    def fetch_and_cache(self, season: int, views: list[str], current_season: int | None = None) -> dict:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        cached = RAW_DIR / f"{season}_{'_'.join(views)}.json"
        if cached.exists():
            return json.loads(cached.read_text())
        data = self.fetch_season(season, views, current_season)
        cached.write_text(json.dumps(data))
        return data

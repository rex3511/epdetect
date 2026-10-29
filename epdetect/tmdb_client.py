"""TMDB API client (v3, bearer token) for episode metadata, with disk cache."""
import json
import os
import time
from pathlib import Path
from typing import Optional

import httpx


class TMDBClient:
    def __init__(self, read_access_token: str, cache_dir: Optional[str] = None):
        cache_dir = cache_dir or os.getenv("TMDB_CACHE_DIR", "/data/.tmdb_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(
            base_url="https://api.themoviedb.org/3",
            headers={
                "Authorization": f"Bearer {read_access_token}",
                "accept": "application/json",
            },
            timeout=30.0,
        )

    def _get_cache_path(self, show_name: str, season: int) -> Path:
        safe = "".join(c for c in show_name if c.isalnum() or c in " -").strip()
        return self.cache_dir / f"{safe}_s{season}.json"

    def _search_show(self, show_name: str) -> Optional[dict]:
        r = self.client.get("/search/tv", params={"query": show_name})
        r.raise_for_status()
        results = r.json().get("results", [])
        return results[0] if results else None

    def get_season_episodes(self, show_name: str, season: int) -> list[dict]:
        cache_path = self._get_cache_path(show_name, season)
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        show = self._search_show(show_name)
        if not show:
            raise ValueError(f"Show not found: {show_name}")

        time.sleep(0.25)
        sid = show["id"]
        r = self.client.get(f"/tv/{sid}/season/{season}")
        r.raise_for_status()
        data = r.json()

        episodes = []
        for ep in data.get("episodes", []):
            episodes.append(
                {
                    "season": season,
                    "episode": ep["episode_number"],
                    "name": ep.get("name", "Unknown"),
                    "overview": ep.get("overview") or "",
                }
            )

        cache_path.write_text(json.dumps(episodes, indent=1))
        return episodes

    def format_episode(self, show_name: str, episode: dict) -> str:
        season = episode.get("season", 1)
        ep_num = episode.get("episode", 0)
        title = episode.get("name", "Unknown")
        return f"S{season:02d}E{ep_num:02d} - {title}"

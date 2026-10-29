"""Episode detection pipeline:

1. Extract English subtitle track (mkvextract) / parse SRT
2. Summarize with a local LM Studio chat model
3. Embed summary + episode overviews (local LM Studio embedding model)
4. Rank candidates by cosine similarity, take top-K
5. Clef (local decision model, /v1/systemone) picks among top-K
6. Jev (OpenRouter) arbitrates low-confidence / close-margin cases
"""
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

import httpx
import pysrt


class EpisodeDetector:
    def __init__(
        self,
        lm_studio_url: str = "http://host.docker.internal:1234",
        summary_model: str = "google/gemma-4-e4b",
        embedding_model: str = "text-embedding-nomic-embed-text-v1.5",
        clef_model: str = "cloudflare_clef-flash",
        openrouter_api_key: str = "",
        jev_model: str = "typesafe/jev-router",
        top_k: int = 6,
        clef_conf_gate: float = 0.85,
        clef_margin_gate: float = 0.15,
        jev_full_season_below: float = 0.30,
    ):
        self.lm_studio_url = lm_studio_url.rstrip("/")
        self.summary_model = summary_model
        self.embedding_model = embedding_model
        self.clef_model = clef_model
        self.openrouter_api_key = openrouter_api_key
        self.jev_model = jev_model
        self.top_k = top_k
        self.clef_conf_gate = clef_conf_gate
        self.clef_margin_gate = clef_margin_gate
        self.jev_full_season_below = jev_full_season_below
        self.client = httpx.Client(timeout=600.0)

    # ---------------- subtitle extraction ----------------

    def extract_subtitle(self, file_path: str) -> Optional[str]:
        path = Path(file_path)

        if path.suffix.lower() == ".srt":
            return file_path

        if path.suffix.lower() != ".mkv":
            raise ValueError(f"Unsupported file type: {path.suffix}")

        srt_path = path.with_suffix(".srt")
        if srt_path.exists():
            return str(srt_path)

        track_id = self._find_english_track(str(path))
        if track_id is None:
            return None

        result = subprocess.run(
            ["mkvextract", "tracks", str(path), f"{track_id}:{str(srt_path)}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"mkvextract failed: {result.stderr}")

        return str(srt_path)

    def _find_english_track(self, mkv_path: str) -> Optional[int]:
        result = subprocess.run(
            ["mkvmerge", "-J", mkv_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None

        try:
            info = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None

        for track in info.get("tracks", []):
            if track.get("type") == "subtitles":
                props = track.get("properties", {})
                lang = props.get("language", "").lower()
                if lang in ("eng", "english"):
                    return track.get("id")

        return None

    _HTML_TAG = re.compile(r"<[^>]+>")
    _NONALPHA = re.compile(r"[^a-zA-Z0-9 ,.'?!\"'-]")

    def extract_subtitle_text(self, subtitle_path: str, char_cap: int = 12000) -> str:
        subs = list(pysrt.open(subtitle_path, encoding="utf-8"))
        parts = []
        for sub in subs:
            t = self._HTML_TAG.sub("", sub.text)
            t = self._NONALPHA.sub(" ", t)
            t = re.sub(r"\s+", " ", t).strip()
            if t:
                parts.append(t)
        return " ".join(parts)[:char_cap]

    # ---------------- local models (LM Studio) ----------------

    def summarize(self, text: str) -> str:
        prompt = (
            "Below is the cleaned subtitle text of a TV episode. "
            "Write a single-paragraph plot summary (max 150 words). Include key plot points, "
            "character names, and specific events that uniquely identify this episode. "
            "Do not mention that this is subtitle text.\n\n"
            f"{text}\n\nSummary:"
        )
        response = self.client.post(
            f"{self.lm_studio_url}/v1/chat/completions",
            json={
                "model": self.summary_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500,
                "temperature": 0.3,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.post(
            f"{self.lm_studio_url}/v1/embeddings",
            json={"model": self.embedding_model, "input": texts},
        )
        response.raise_for_status()
        return [d["embedding"] for d in response.json()["data"]]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        ma = sum(x * x for x in a) ** 0.5
        mb = sum(x * x for x in b) ** 0.5
        return dot / (ma * mb) if ma and mb else 0.0

    def clef_decide(self, summary: str, cand_eps: list[dict], show_name: str) -> dict:
        criteria = {}
        for e in cand_eps:
            lab = e["label"]
            criteria[lab] = f"{e['name']}: {e.get('overview', '')}"[:110]
        payload = {
            "model": self.clef_model,
            "state": f"Episode plot summary ({show_name}):\n{summary[:450]}",
            "questions": {
                "episode": {
                    "type": "choice",
                    "instructions": "Which episode is this? Pick the best match by plot details.",
                    "criteria": criteria,
                }
            },
        }
        response = self.client.post(
            f"{self.lm_studio_url}/v1/systemone", json=payload
        )
        response.raise_for_status()
        return response.json()["answers"]["episode"]

    def jev_pick(self, summary: str, candidates: list[dict], retries: int = 2) -> tuple[dict, float]:
        cand_text = "\n".join(
            f"- {e['label']} {e['name']}: {e.get('overview', '')}" for e in candidates
        )
        prompt = (
            "You are matching a TV episode plot summary to the correct episode of a TV show. "
            "The summary is auto-generated from subtitles and is imperfect/vague — pick the BEST available "
            "match among the candidates by plot details and character names, even if the fit is not perfect. "
            "Only answer NONE if nothing is even plausibly related.\n\n"
            f"PLOT SUMMARY (from subtitles):\n{summary}\n\n"
            f"CANDIDATE EPISODES:\n{cand_text}\n\n"
            'Return ONLY JSON: {"episode": "SxxEyy", "reason": "one short sentence"}.'
        )
        last_text = ""
        for _ in range(retries + 1):
            response = self.client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={
                    "model": self.jev_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                },
                headers={"Authorization": f"Bearer {self.openrouter_api_key}"},
            )
            response.raise_for_status()
            result = response.json()
            text = result["choices"][0]["message"].get("content") or ""
            last_text = text
            cost = result.get("usage", {}).get("cost") or 0.0
            if not text.strip():
                continue
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                try:
                    return json.loads(m.group(0)), cost
                except json.JSONDecodeError:
                    pass
        return {"episode": "PARSE_FAIL", "reason": last_text[:120]}, cost

    # ---------------- pipeline ----------------

    @staticmethod
    def _label(ep: dict) -> str:
        return f"S{ep['season']:02d}E{ep['episode']:02d}"

    def detect(
        self,
        file_path: str,
        episodes: list[dict],
        season: Optional[int] = None,
        show_name: str = "TV show",
    ) -> dict:
        """Run the full pipeline. Returns a result dict with match, confidence,
        stage info, and jev arbitration details."""
        srt_path = self.extract_subtitle(file_path)
        if srt_path is None:
            return {"matched_episode": None, "confidence": 0.0, "action": "no_english_subtitle"}

        raw_text = self.extract_subtitle_text(srt_path)
        summary = self.summarize(raw_text)

        # normalize labels onto episodes
        pool = []
        for ep in episodes:
            e = dict(ep)
            e["label"] = self._label(e)
            pool.append(e)
        if season is not None:
            season_pool = [e for e in pool if e["season"] == season]
            if season_pool:
                pool = season_pool

        # embedding pre-filter
        ep_texts = [f"{e['name']}: {e.get('overview', '')}" for e in pool]
        ep_embs = self.embed(ep_texts)
        (sum_emb,) = self.embed([summary])
        ranked = sorted(
            zip(pool, ep_embs),
            key=lambda t: -self.cosine_similarity(sum_emb, t[1]),
        )
        candidates = [e for e, _ in ranked[: self.top_k]]

        # clef decision
        ans = self.clef_decide(summary, candidates, show_name)
        probs = ans.get("probabilities") or {}
        sorted_probs = sorted(probs.items(), key=lambda kv: -kv[1])
        clef_choice = ans["choice"]
        clef_conf = ans.get("confidence") or 0.0

        # escalation check
        escalate = clef_conf < self.clef_conf_gate
        why = f"conf {clef_conf:.2f} < {self.clef_conf_gate}" if escalate else ""
        if not escalate and len(sorted_probs) >= 2:
            margin = sorted_probs[0][1] - sorted_probs[1][1]
            if margin < self.clef_margin_gate:
                escalate = True
                why = f"margin {margin:.2f} < {self.clef_margin_gate}"

        result = {
            "matched_episode": None,
            "confidence": clef_conf,
            "clef_choice": clef_choice,
            "jev_choice": None,
            "jev_reason": None,
            "escalated": False,
            "summary": summary,
        }

        if escalate and self.openrouter_api_key:
            result["escalated"] = True
            if clef_conf < self.jev_full_season_below:
                jev_cands = pool  # very unsure: full season
            else:
                jev_cands = candidates
            jans, _cost = self.jev_pick(summary, jev_cands)
            jev_choice = jans.get("episode")
            result["jev_choice"] = jev_choice
            result["jev_reason"] = jans.get("reason")
            final = jev_choice if jev_choice and jev_choice not in ("NONE", "PARSE_FAIL") else clef_choice
        else:
            result["escalation_reason"] = why or None
            final = clef_choice

        match = next((e for e in pool if e["label"] == final), None)
        result["matched_episode"] = match
        return result

    def close(self):
        self.client.close()

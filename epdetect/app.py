from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import re
import shutil
from pathlib import Path
from detector import EpisodeDetector
from tmdb_client import TMDBClient

app = FastAPI()

tmdb_token = os.getenv("TMDB_READ_ACCESS_TOKEN", "")
lm_studio_url = os.getenv("LM_STUDIO_URL", "http://host.docker.internal:1234")
summary_model = os.getenv("SUMMARY_MODEL", "google/gemma-4-e4b")
embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")
clef_model = os.getenv("CLEF_MODEL", "cloudflare_clef-flash")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
jev_model = os.getenv("JEV_MODEL", "typesafe/jev-router")
confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.85"))
output_dir = os.getenv("OUTPUT_DIR", "")

tmdb_client = TMDBClient(tmdb_token) if tmdb_token else None
detector = EpisodeDetector(
    lm_studio_url=lm_studio_url,
    summary_model=summary_model,
    embedding_model=embedding_model,
    clef_model=clef_model,
    openrouter_api_key=openrouter_api_key,
    jev_model=jev_model,
)


class DetectRequest(BaseModel):
    show_name: str
    season: int
    subtitle_path: str
    rename: bool = True


class DetectResponse(BaseModel):
    original_file: str
    matched_episode: Optional[dict]
    confidence: float
    action: str
    new_filename: Optional[str]
    renamed_path: Optional[str] = None
    clef_choice: Optional[str] = None
    jev_choice: Optional[str] = None
    jev_reason: Optional[str] = None
    escalated: bool = False
    summary: Optional[str] = None


class BatchRequest(BaseModel):
    show_name: str
    season: Optional[int] = None  # omit to auto-detect per file from SxxEyy in filename
    folder_path: str
    extensions: list[str] = ["srt"]
    rename: bool = True
    dry_run: bool = False
    output_dir: Optional[str] = None


class BatchResponse(BaseModel):
    results: list[DetectResponse]
    summary: dict


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "tmdb_configured": bool(tmdb_token),
        "jev_configured": bool(openrouter_api_key),
    }


@app.get("/models")
async def list_models():
    import httpx

    try:
        resp = httpx.get(f"{lm_studio_url}/v1/models", timeout=10.0)
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _rename(src_path: str, show_name: str, episode: dict, ext: str,
            dest_dir: Optional[str] = None) -> tuple[str, str]:
    ep_info = tmdb_client.format_episode(show_name, episode)
    new_filename = f"{show_name} - {ep_info}{ext}"
    src = Path(src_path)
    if dest_dir:
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        dst = Path(dest_dir) / new_filename
    else:
        dst = src.parent / new_filename
    shutil.move(str(src), str(dst))
    return new_filename, str(dst)


@app.post("/detect", response_model=DetectResponse)
async def detect_episode(request: DetectRequest):
    if not tmdb_client:
        raise HTTPException(status_code=500, detail="TMDB_READ_ACCESS_TOKEN not configured")

    try:
        episodes = tmdb_client.get_season_episodes(request.show_name, request.season)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch TMDB data: {e}")

    try:
        result = detector.detect(
            request.subtitle_path, episodes,
            season=request.season, show_name=request.show_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

    original_filename = Path(request.subtitle_path).name
    matched = result.get("matched_episode")
    confidence = result.get("confidence", 0.0)

    if matched is None and result.get("action") == "no_english_subtitle":
        return DetectResponse(
            original_file=original_filename,
            matched_episode=None,
            confidence=0.0,
            action="no_english_subtitle",
            new_filename=None,
        )

    action = "skipped"
    new_filename = None
    renamed_path = None

    if matched is not None and confidence >= confidence_threshold:
        action = "auto_renamed" if request.rename else "matched"
        if request.rename:
            try:
                new_filename, renamed_path = _rename(
                    request.subtitle_path, request.show_name, matched,
                    Path(request.subtitle_path).suffix,
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to rename file: {e}")
    elif matched is not None:
        # With clef/jev the decision itself is the match; confidence drives action
        action = "needs_confirmation"

    return DetectResponse(
        original_file=original_filename,
        matched_episode=matched,
        confidence=confidence,
        action=action,
        new_filename=new_filename,
        renamed_path=renamed_path,
        clef_choice=result.get("clef_choice"),
        jev_choice=result.get("jev_choice"),
        jev_reason=result.get("jev_reason"),
        escalated=result.get("escalated", False),
        summary=result.get("summary"),
    )


_SEASON_RE = re.compile(r"[Ss](\d{1,2})[Ee]\d{1,3}")


def _detect_season(filename: str) -> Optional[int]:
    m = _SEASON_RE.search(filename)
    return int(m.group(1)) if m else None


def _process_one(sub_file: Path, request: BatchRequest, season: int,
                 episodes: list[dict]) -> DetectResponse:
    try:
        result = detector.detect(
            str(sub_file), episodes,
            season=season, show_name=request.show_name,
        )
        matched = result.get("matched_episode")
        confidence = result.get("confidence", 0.0)

        action = "skipped"
        new_filename = None
        renamed_path = None

        if request.dry_run:
            action = "dry_run"
            if matched is not None:
                ep_info = tmdb_client.format_episode(request.show_name, matched)
                new_filename = f"{request.show_name} - {ep_info}{sub_file.suffix}"
        elif matched is not None and confidence >= confidence_threshold and request.rename:
            action = "auto_renamed"
            try:
                new_filename, renamed_path = _rename(
                    str(sub_file), request.show_name, matched, sub_file.suffix,
                    dest_dir=request.output_dir or output_dir or None,
                )
            except Exception as e:
                action = f"error: rename failed - {e}"
        elif matched is not None:
            action = "needs_confirmation"

        return DetectResponse(
            original_file=sub_file.name,
            matched_episode=matched,
            confidence=confidence,
            action=action,
            new_filename=new_filename,
            renamed_path=renamed_path,
            clef_choice=result.get("clef_choice"),
            jev_choice=result.get("jev_choice"),
            jev_reason=result.get("jev_reason"),
            escalated=result.get("escalated", False),
            summary=result.get("summary"),
        )
    except Exception as e:
        return DetectResponse(
            original_file=sub_file.name,
            matched_episode=None,
            confidence=0.0,
            action=f"error: {str(e)}",
            new_filename=None,
        )


@app.post("/batch", response_model=BatchResponse)
async def batch_detect(request: BatchRequest):
    if not tmdb_client:
        raise HTTPException(status_code=500, detail="TMDB_READ_ACCESS_TOKEN not configured")

    folder = Path(request.folder_path)
    subtitle_files = []
    for ext in request.extensions:
        subtitle_files.extend(folder.rglob(f"*.{ext}"))

    # Group files into per-season runs. An explicit season applies to all files;
    # otherwise detect per file from SxxEyy in the name.
    groups: dict[Optional[int], list[Path]] = {}
    for f in sorted(subtitle_files):
        s = request.season if request.season is not None else _detect_season(f.name)
        groups.setdefault(s, []).append(f)

    results: list[DetectResponse] = []
    episodes_cache: dict[int, list[dict]] = {}

    for season in sorted(groups, key=lambda x: (x is None, x)):
        files = groups[season]
        if season is None:
            for f in files:
                results.append(DetectResponse(
                    original_file=f.name,
                    matched_episode=None,
                    confidence=0.0,
                    action="skipped: could not determine season from filename",
                    new_filename=None,
                ))
            continue

        if season not in episodes_cache:
            try:
                episodes_cache[season] = tmdb_client.get_season_episodes(
                    request.show_name, season
                )
            except Exception as e:
                for f in files:
                    results.append(DetectResponse(
                        original_file=f.name,
                        matched_episode=None,
                        confidence=0.0,
                        action=f"error: TMDB fetch failed for season {season}: {e}",
                        new_filename=None,
                    ))
                continue

        for sub_file in files:
            results.append(_process_one(sub_file, request, season, episodes_cache[season]))

    return BatchResponse(
        results=results,
        summary={
            "total": len(results),
            "seasons": {str(s): len(fs) for s, fs in groups.items()},
            "auto_renamed": sum(1 for r in results if r.action == "auto_renamed"),
            "needs_confirmation": sum(
                1 for r in results if r.action == "needs_confirmation"
            ),
            "skipped": sum(1 for r in results if r.action.startswith("skipped")),
            "escalated": sum(1 for r in results if r.escalated),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

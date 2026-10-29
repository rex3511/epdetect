# epdetect

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/L6X129OAZT)

A local AI-powered service that identifies and renames unlabeled TV episodes from their subtitle files.

## How it works

For each video/subtitle file:

1. **Extract** the English subtitle track from MKV (via `mkvextract`), or read an `.srt` directly
2. **Summarize** the subtitle text with a local LM Studio chat model
3. **Embed** the summary and every TMDB episode overview for the season (local embedding model)
4. **Rank** episodes by cosine similarity and take the top 6 candidates
5. **Clef** (local decision model) picks the best match among the candidates, with a confidence score
6. **Jev** (OpenRouter) arbitrates when clef is unsure; very unsure escalates to a full-season jev pick

Batch runs group files into **per-season runs**. Each file is only ever compared against episodes from its own season, which is what drives accuracy.

## Prerequisites

- Docker
- LM Studio running locally with three models loaded:
  - a chat model for summaries (default `google/gemma-4-e4b`)
  - an embedding model (default `text-embedding-nomic-embed-text-v1.5`)
  - the clef decision model (default `cloudflare_clef-flash`)
- TMDB Read Access Token (https://www.themoviedb.org/settings/api)
- OpenRouter API key (for jev arbitration)

## Setup

1. Copy the environment file and fill in your keys:
   ```bash
   cp .env.example .env
   ```

2. Build and start:
   ```bash
   docker compose up -d
   ```

3. Health check:
   ```bash
   curl http://localhost:8000/health
   ```

## Usage

All interaction is over the REST API (e.g. with `curl`).

### Detect a single file

`/detect` requires an explicit `season`:

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "show_name": "The Office",
    "season": 1,
    "subtitle_path": "/data/The.Office.S01E01.mkv",
    "rename": true
  }'
```

### Batch process a folder

Omit `season` to auto-detect each file's season from the `SxxEyy` in its filename — files are grouped into per-season runs. Files with no recognizable season are skipped.

```bash
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{
    "show_name": "The Office",
    "folder_path": "/data/the-office",
    "dry_run": true
  }'
```

Set `"dry_run": false` (and `"rename": true`, the default) to actually rename.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service + key configuration status |
| `GET` | `/models` | List models loaded in LM Studio |
| `POST` | `/detect` | Detect a single file (season required) |
| `POST` | `/batch` | Process a folder, grouped per season |

## Configuration

Set in `.env` (loaded by docker compose):

| Variable | Default | Description |
|----------|---------|-------------|
| `TMDB_READ_ACCESS_TOKEN` | (required) | TMDB v3 read access token |
| `OPENROUTER_API_KEY` | (required) | OpenRouter key for jev arbitration |
| `LM_STUDIO_URL` | `http://host.docker.internal:1234` | LM Studio base URL |
| `SUMMARY_MODEL` | `google/gemma-4-e4b` | LM Studio chat model for summaries |
| `EMBEDDING_MODEL` | `text-embedding-nomic-embed-text-v1.5` | LM Studio embedding model |
| `CLEF_MODEL` | `cloudflare_clef-flash` | Local decision model |
| `JEV_MODEL` | `typesafe/jev-router` | OpenRouter arbitration model |
| `CONFIDENCE_THRESHOLD` | `0.85` | Min clef confidence to auto-rename |
| `TMDB_CACHE_DIR` | `/data/.tmdb_cache` | Episode metadata cache location |
| `DATA_DIR` | `./data` | Media files volume mount (host side) |
| `OUTPUT_DIR` | `./output` | Optional output folder for renamed files |

The matching thresholds (top-6 candidates, jev escalation below 0.85 confidence or <0.15 margin, full-season jev below 0.30) are constructor defaults in `detector.py` and not currently env-configurable.

## Response fields

Each detection returns the match plus pipeline diagnostics:

- `matched_episode` / `confidence` — the match and clef's confidence
- `action` — `auto_renamed`, `needs_confirmation`, `skipped`, `dry_run`, or an `error:`/`no_english_subtitle` reason
- `clef_choice` — local model's pick
- `escalated` — whether jev was called
- `jev_choice` / `jev_reason` — arbitration result
- `summary` — the generated plot summary
- `new_filename` / `renamed_path` — when renaming occurred (or would, in dry-run)

## How matches are decided

| Condition | Outcome |
|-----------|---------|
| clef confidence ≥ `CONFIDENCE_THRESHOLD` (0.85) | Clef's pick, auto-rename |
| clef confidence < 0.85, or top-2 margin < 0.15 | Jev arbitration among the top 6 |
| clef confidence < 0.30 | Jev picks from the **full season** |
| jev returns `NONE` / unparseable | Falls back to clef's pick |
| Final confidence below threshold | `needs_confirmation` (not renamed) |
| No English subtitle / no match | `skipped` |

## Stopping

```bash
docker compose down
```

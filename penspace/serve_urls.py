"""Reference presigned-playback-URL service.

The app never talks to a GPU. It asks this endpoint for a short-lived URL and
hands that straight to the audio player, which streams the object from S3 with
range requests (so seeking works without downloading the whole file).

This is a reference implementation meant to be folded into your existing
Penspace backend -- swap the JSONL catalog for your real summaries table and
apply your own auth/entitlement check where marked.

    pip install fastapi "uvicorn[standard]"
    PENSPACE_S3_BUCKET=penspace-audio PENSPACE_CATALOG=./penspace_out/catalog.jsonl \
        uvicorn penspace.serve_urls:app --port 8080
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException

from .config import Config
from .storage import S3Storage

app = FastAPI(title="Penspace audio URLs")

cfg = Config.from_env()
storage = S3Storage(cfg)


def _load_catalog() -> Dict[str, dict]:
    path = Path(os.environ.get("PENSPACE_CATALOG", "./penspace_out/catalog.jsonl"))
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


CATALOG = _load_catalog()


@app.get("/healthz")
def healthz():
    return {"ok": True, "summaries": len(CATALOG)}


@app.get("/summaries/{summary_id}/audio")
def audio_url(summary_id: str):
    row = CATALOG.get(summary_id)
    if not row or not row.get("audio_key"):
        raise HTTPException(status_code=404, detail="summary not rendered")

    # TODO: your entitlement check goes here, before minting the URL.

    return {
        "url": storage.presign(row["audio_key"]),
        "expires_in": cfg.presign_ttl,
        "duration": row.get("duration"),
        "render_id": row.get("render_id"),
    }


@app.get("/summaries/{summary_id}/timing")
def timing_url(summary_id: str):
    """Chunk timing map, for sentence highlighting and scrub-to-paragraph."""
    row = CATALOG.get(summary_id)
    if not row or not row.get("timing_key"):
        raise HTTPException(status_code=404, detail="summary not rendered")
    return {"url": storage.presign(row["timing_key"]), "expires_in": cfg.presign_ttl}

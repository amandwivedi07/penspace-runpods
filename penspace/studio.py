"""Browser studio for GPU rendering and playback.

Run on the pod:

    bash penspace/start_studio.sh

Then open the Web URL from the dashboard, or tunnel:

    ssh -L 8080:localhost:8080 -p <port> root@<host>
    open http://localhost:8080
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .ingest import ingest, slugify
from .runner import SummaryJob, load_manifest

log = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

WORKSPACE = Path(os.environ.get("PENSPACE_WORKSPACE", "/workspace"))
MANIFEST_DIR = Path(os.environ.get("PENSPACE_MANIFEST_DIR", WORKSPACE / "manifests"))
WORK_DIR = Path(os.environ.get("PENSPACE_WORK_DIR", WORKSPACE / "out"))
UPLOAD_DIR = Path(os.environ.get("PENSPACE_UPLOAD_DIR", WORKSPACE / "uploads"))
SUMMARY_DIR = Path(os.environ.get("PENSPACE_SUMMARY_DIR", WORKSPACE / "summaries"))
STUDIO_MANIFEST = MANIFEST_DIR / "studio.jsonl"
HTML_PATH = Path(__file__).with_name("studio.html")
ALLOWED_SUFFIXES = {".docx", ".md", ".markdown", ".txt"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Penspace Studio")


@dataclass
class JobState:
    status: str = "idle"  # idle | rendering | done | error
    message: str = ""
    error: Optional[str] = None
    started_at: float = 0.0


_jobs: Dict[str, JobState] = {}
_lock = threading.Lock()
_gpu_lock = threading.Lock()


def _discover_jobs() -> Dict[str, SummaryJob]:
    jobs: Dict[str, SummaryJob] = {}
    if not MANIFEST_DIR.exists():
        return jobs
    for path in sorted(MANIFEST_DIR.glob("*.jsonl")):
        if path.name.startswith("._"):
            continue
        try:
            for job in load_manifest(path):
                jobs[job.id] = job
        except (OSError, ValueError) as exc:
            log.warning("skipping manifest %s: %s", path, exc)
    return jobs


def _unique_id(base: str) -> str:
    taken = set(_discover_jobs())
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def _start_render(summary_id: str):
    jobs = _discover_jobs()
    if summary_id not in jobs:
        return
    with _lock:
        if _jobs.get(summary_id, JobState()).status == "rendering":
            return
    thread = threading.Thread(target=_run_render, args=(jobs[summary_id],), daemon=True)
    thread.start()


def _latest_render(summary_id: str) -> Optional[Path]:
    base = WORK_DIR / summary_id
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and (p / "audio.m4a").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _chapter_row(job: SummaryJob) -> dict:
    render_dir = _latest_render(job.id)
    timing_path = render_dir / "timing.json" if render_dir else None
    duration = chunk_count = None
    status = "missing"
    if timing_path and timing_path.exists():
        timing = json.loads(timing_path.read_text())
        duration = timing.get("duration")
        chunk_count = len(timing.get("chunks", []))
        status = "ready"

    with _lock:
        job_state = _jobs.get(job.id, JobState())

    return {
        "id": job.id,
        "title": job.title or job.id,
        "author": job.author,
        "status": status,
        "duration": duration,
        "chunk_count": chunk_count,
        "render_id": render_dir.name if render_dir else None,
        "job_status": job_state.status if job_state.status != "idle" else None,
        "job_message": job_state.message,
        "job_error": job_state.error,
    }


def _run_render(job: SummaryJob):
    with _lock:
        _jobs[job.id] = JobState(
            status="rendering",
            message="Waiting for GPU…",
            started_at=time.time(),
        )

    try:
        with _gpu_lock:
            with _lock:
                _jobs[job.id] = JobState(
                    status="rendering",
                    message="Loading model and synthesizing on GPU…",
                    started_at=time.time(),
                )
            WORK_DIR.mkdir(parents=True, exist_ok=True)
            job_dir = WORKSPACE / "studio-jobs"
            job_dir.mkdir(parents=True, exist_ok=True)
            manifest = job_dir / f"{job.id}.jsonl"
            row = {
                "id": job.id,
                "title": job.title,
                "author": job.author,
                "text": job.text,
            }
            manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n")

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            cmd = [
                sys.executable,
                "-m",
                "penspace.cli",
                "render",
                "--manifest",
                str(manifest),
                "--work-dir",
                str(WORK_DIR),
            ]
            log.info("render subprocess: %s", " ".join(cmd))
            proc = subprocess.run(cmd, cwd=str(WORKSPACE), env=env)
            if proc.returncode not in (0, 2):
                raise RuntimeError(f"render exited {proc.returncode}")
            if not _latest_render(job.id):
                raise RuntimeError("render finished but audio.m4a is missing")
            started = _jobs.get(job.id, JobState()).started_at or time.time()
            msg = f"Done in {time.time() - started:.0f}s"
            if proc.returncode == 2:
                msg += " (some chunks failed QA)"
            with _lock:
                _jobs[job.id] = JobState(status="done", message=msg)
    except Exception as exc:  # noqa: BLE001 — surface to UI
        log.exception("render failed for %s", job.id)
        with _lock:
            _jobs[job.id] = JobState(status="error", error=str(exc), message="Render failed")


@app.get("/", response_class=HTMLResponse)
def index():
    if not HTML_PATH.exists():
        raise HTTPException(status_code=500, detail="studio.html missing")
    return HTML_PATH.read_text()


@app.get("/api/chapters")
def list_chapters():
    rows = [_chapter_row(job) for job in _discover_jobs().values()]
    return sorted(rows, key=lambda r: (r.get("title") or r["id"]).lower())


@app.get("/api/chapters/{summary_id}")
def get_chapter(summary_id: str):
    jobs = _discover_jobs()
    if summary_id not in jobs:
        raise HTTPException(status_code=404, detail="chapter not found")
    return _chapter_row(jobs[summary_id])


@app.post("/api/chapters/{summary_id}/render")
def render_chapter(summary_id: str):
    jobs = _discover_jobs()
    if summary_id not in jobs:
        raise HTTPException(status_code=404, detail="chapter not found")
    _start_render(summary_id)
    return {"ok": True, "message": "render started"}


@app.post("/api/upload")
async def upload_chapter(
    file: UploadFile = File(...),
    title: str = Form(""),
    author: str = Form(""),
    auto_render: str = Form("true"),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type. Use: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="file too large (max 10 MB)")

    stem = Path(file.filename or "summary").stem
    chapter_title = title.strip() or stem.lstrip("_").strip()
    summary_id = _unique_id(slugify(chapter_title or stem))

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    source = UPLOAD_DIR / f"{summary_id}{suffix}"
    source.write_bytes(raw)

    try:
        result = ingest(
            source=source,
            out_dir=SUMMARY_DIR,
            summary_id=summary_id,
            title=chapter_title,
            author=author.strip() or None,
            manifest=STUDIO_MANIFEST,
        )
    except Exception as exc:  # noqa: BLE001
        source.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Ensure manifest points at summaries with a stable relative path.
    rows = []
    if STUDIO_MANIFEST.exists():
        rows = [
            json.loads(line)
            for line in STUDIO_MANIFEST.read_text().splitlines()
            if line.strip()
        ]
    for row in rows:
        if row.get("id") == summary_id:
            row["text_file"] = f"../summaries/{summary_id}.md"
    STUDIO_MANIFEST.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    )

    jobs = _discover_jobs()
    if summary_id not in jobs:
        raise HTTPException(status_code=500, detail="ingest succeeded but chapter missing")

    if auto_render.lower() in ("1", "true", "yes", "on"):
        _start_render(summary_id)

    chapter = _chapter_row(jobs[summary_id])
    return {"ok": True, "chapter": chapter, "ingest": result}


@app.get("/api/chapters/{summary_id}/audio")
def chapter_audio(summary_id: str):
    render_dir = _latest_render(summary_id)
    if not render_dir:
        raise HTTPException(status_code=404, detail="not rendered")
    path = render_dir / "audio.m4a"
    if not path.exists():
        raise HTTPException(status_code=404, detail="audio missing")
    return FileResponse(path, media_type="audio/mp4", filename=f"{summary_id}.m4a")


@app.get("/api/chapters/{summary_id}/timing")
def chapter_timing(summary_id: str):
    render_dir = _latest_render(summary_id)
    if not render_dir:
        raise HTTPException(status_code=404, detail="not rendered")
    path = render_dir / "timing.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="timing missing")
    return json.loads(path.read_text())


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "workspace": str(WORKSPACE),
        "chapters": len(_discover_jobs()),
        "gpu": os.environ.get("PENSPACE_DEVICE", "auto"),
    }

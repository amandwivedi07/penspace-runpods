"""RunPod Serverless entrypoint for the Penspace narration renderer.

The renderer itself is unchanged: this module is a thin adapter that turns a
serverless request into `SummaryJob`s, hands them to the same `Runner` the CLI
uses, and reports back where the audio landed in S3.

Two things about this workload shape the design.

**The model is the expensive part, not the request.** Loading Qwen3-TTS plus
the Whisper QA model costs roughly a minute and several GB of reads. RunPod
bills that on every cold start, so the runner is built once at *import* time —
worker init — rather than inside `handler()`. A warm worker answers the next
request with the weights already resident.

**Therefore: send batches.** A request may carry one summary or a hundred
(`{"jobs": [...]}`). One request with fifty summaries pays the cold start once;
fifty requests may pay it fifty times. For the backfill this is the difference
between a sensible bill and a silly one — see DEPLOY_SERVERLESS.md.

Retries are safe by construction. `Runner.render` fingerprints the normalized
text and the voice into a `render_id` and skips the work when that object is
already in S3, so a re-delivered request returns the existing key instead of
re-rendering and re-billing.
"""

from __future__ import annotations

import logging
import os
import traceback
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(
    level=os.environ.get("PENSPACE_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("penspace.handler")

# Imports, guarded and logged.
#
# These pull in torch, the Qwen runtime and the RunPod SDK, and any of them can
# fail in ways that kill the process before a single useful line is written —
# a CUDA architecture the wheels have no kernels for will abort rather than
# raise. RunPod then reports a bare "worker exited with exit code 1", which is
# indistinguishable from a missing package, a bad build, or an OOM.
#
# Naming the failing import and re-raising costs nothing and turns a silent
# crash-loop into one log line that says what to fix.
try:
    import runpod

    from .config import Config
    from .runner import Runner, SummaryJob, prepare_clone
    from .storage import S3Storage
except BaseException as exc:  # noqa: BLE001 — includes SystemExit/abort paths
    log.critical(
        "IMPORT FAILED: %s: %s\n%s",
        type(exc).__name__,
        exc,
        traceback.format_exc(),
    )
    raise

# A CUDA mismatch usually surfaces the moment a device is touched, not at
# import, so probe it here where the result can still be logged.
try:
    import torch

    if torch.cuda.is_available():
        log.info(
            "GPU: %s (capability %s), torch %s, cuda %s",
            torch.cuda.get_device_name(0),
            ".".join(map(str, torch.cuda.get_device_capability(0))),
            torch.__version__,
            torch.version.cuda,
        )
    else:
        log.warning("No CUDA device visible — the render will fall back to CPU and crawl.")
except BaseException as exc:  # noqa: BLE001
    log.critical("CUDA PROBE FAILED: %s: %s", type(exc).__name__, exc)

WORK_DIR = Path(os.environ.get("PENSPACE_WORK_DIR", "/work"))
# A remote text file should not be able to hang the worker forever.
FETCH_TIMEOUT = int(os.environ.get("PENSPACE_FETCH_TIMEOUT", "30"))
# ~2 MB of text is far beyond any single summary (the largest in the catalogue
# is around 40 KB), so this only ever catches a mistake.
MAX_TEXT_BYTES = int(os.environ.get("PENSPACE_MAX_TEXT_BYTES", str(2 * 1024 * 1024)))


# --------------------------------------------------------------------------
# Worker init — runs once, at import, before the first request is accepted.
# --------------------------------------------------------------------------

def _build_runner() -> Runner:
    cfg = prepare_clone(Config.from_env())

    if cfg.is_clone and not os.environ.get("PENSPACE_REF_TEXT"):
        # prepare_clone() will have just transcribed the reference with Whisper.
        # That transcript is part of the render fingerprint, so if it comes out
        # even slightly different on the next worker, every summary rendered
        # there gets a NEW render_id — the S3 skip stops matching and the whole
        # backfill silently renders twice. Pin it. See DEPLOY_RUNPOD.md §4.
        log.warning(
            "PENSPACE_REF_TEXT is not pinned: the reference transcript was "
            "inferred, and may differ between workers. Different transcript => "
            "different render_id => duplicate renders. Pin it in the endpoint "
            "environment."
        )

    storage = S3Storage(cfg) if cfg.s3_bucket else None
    if storage is None:
        log.warning(
            "No PENSPACE_S3_BUCKET set: audio stays on the worker's local disk "
            "and is lost when it scales down. Set the bucket."
        )
    else:
        # Raising here fails the endpoint at cold start, which is the point: a
        # credential problem found at upload time has already been billed for
        # the render.
        storage.preflight()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    return Runner(cfg, WORK_DIR, storage)


try:
    RUNNER: Runner | None = _build_runner()
    INIT_ERROR: str | None = None
    log.info("worker ready (work_dir=%s)", WORK_DIR)
except Exception as exc:  # noqa: BLE001 — surfaced to every request below
    RUNNER, INIT_ERROR = None, f"{type(exc).__name__}: {exc}"
    log.exception("worker init failed")


# --------------------------------------------------------------------------
# Request parsing
# --------------------------------------------------------------------------

def _text_for(row: Dict[str, Any]) -> str:
    """The summary body, inline or fetched."""
    text = row.get("text")
    if text:
        return str(text)

    url = row.get("text_url")
    if not url:
        raise ValueError("row needs either 'text' or 'text_url'")
    if not str(url).startswith(("http://", "https://")):
        # Anything else — file://, s3://, a bare path — would read the worker's
        # own filesystem on behalf of the caller.
        raise ValueError("'text_url' must be an http(s) URL")

    # Capped. An unbounded read lets one oversized file exhaust a GPU worker's
    # memory, and the failure lands on whatever unrelated job shares the worker.
    with urllib.request.urlopen(str(url), timeout=FETCH_TIMEOUT) as resp:
        body = resp.read(MAX_TEXT_BYTES + 1)
    if len(body) > MAX_TEXT_BYTES:
        raise ValueError(
            f"text at {url} exceeds {MAX_TEXT_BYTES // 1024} KB — "
            "split it into chapters rather than rendering it as one job"
        )
    return body.decode("utf-8")


def _parse(payload: Dict[str, Any]) -> Tuple[List[SummaryJob], List[Dict[str, Any]]]:
    """Split the request into renderable jobs and rows that were malformed.

    A bad row is reported, not raised: one summary with a missing id should not
    throw away the ninety-nine valid ones batched alongside it.
    """
    rows = payload.get("jobs")
    if rows is None:
        rows = [payload]  # single-summary form
    if not isinstance(rows, list):
        raise ValueError("'jobs' must be a list")
    if not rows:
        raise ValueError("no jobs in request")

    jobs: List[SummaryJob] = []
    rejected: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            rejected.append({"index": i, "error": "job must be an object"})
            continue
        try:
            job_id = str(row.get("id") or "").strip()
            if not job_id:
                raise ValueError("row needs an 'id'")
            jobs.append(
                SummaryJob(
                    id=job_id,
                    text=_text_for(row),
                    title=row.get("title"),
                    author=row.get("author"),
                )
            )
        except Exception as exc:  # noqa: BLE001
            rejected.append(
                {"index": i, "id": row.get("id"), "error": f"{type(exc).__name__}: {exc}"}
            )

    return jobs, rejected


# --------------------------------------------------------------------------
# Handler
# --------------------------------------------------------------------------

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    if RUNNER is None:
        # Raised, not returned: this is the worker being broken rather than the
        # request being wrong, so RunPod should mark it FAILED and retry it
        # somewhere else.
        raise RuntimeError(f"worker failed to initialize: {INIT_ERROR}")

    payload = event.get("input") or {}
    jobs, rejected = _parse(payload)

    rendered: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = list(rejected)

    # Rendered one at a time rather than via Runner.run(jobs), so that a summary
    # that fails mid-batch costs only itself: the rest of the batch still lands
    # in S3 and the caller retries the single id.
    for job in jobs:
        try:
            result = RUNNER.render(job)
            row = result.to_dict()
            # `RenderResult` carries `skipped` but `to_dict()` does not emit it,
            # and it is the one field that says whether this request cost GPU
            # time or was served from an existing S3 object. Without it every
            # re-delivered request looks like fresh work, which is exactly
            # backwards for judging a bill.
            row["skipped"] = result.skipped
            rendered.append(row)
            log.info(
                "%s %s (%.1fs audio)",
                "skipped (already rendered)" if result.skipped else "rendered",
                job.id,
                result.duration,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("render failed for %s", job.id)
            failed.append(
                {
                    "id": job.id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=5),
                }
            )

    # Nothing came out, but something went in — fail the whole request so the
    # caller's retry policy sees it, rather than reporting a cheerful 200 over a
    # batch that produced no audio.
    #
    # `jobs or rejected`, not `jobs` alone: rows rejected during parsing never
    # become jobs, so a request in which EVERY row was malformed left `jobs`
    # empty and returned success. The documented contract says an all-failed
    # request fails; this makes that true.
    if not rendered and (jobs or rejected):
        raise RuntimeError(
            f"nothing rendered: {len(jobs)} job(s) failed, "
            f"{len(rejected)} row(s) malformed: {failed[:3]}"
        )

    return {
        "rendered": rendered,
        "failed": failed,
        "counts": {"rendered": len(rendered), "failed": len(failed)},
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

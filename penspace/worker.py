"""Render narration for a Penspace backend, pulling work instead of receiving it.

A rented GPU pod has no inbound address, so nothing can POST to it. This loops
the other way round: ask the backend for queued chapters, render them, post the
results back. The same jobs the admin's Generate button creates.

    python -m penspace.worker --api https://penspace.in/api --token $AUDIO_WORKER_TOKEN

Environment (as for the CLI):
    PENSPACE_S3_BUCKET, AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    PENSPACE_REF_AUDIO, PENSPACE_REF_TEXT   (pin the text — see DEPLOY_RUNPOD.md)

The model loads once, at start, and every chapter after that reuses it. That is
the whole reason this is a long-lived loop rather than a script per chapter.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import Config
from .runner import Runner, SummaryJob, prepare_clone
from .storage import S3Storage

log = logging.getLogger("penspace.worker")

_STOP = False


def _handle_signal(signum, _frame):
    """Finish the chapter in flight, then stop.

    Killing mid-render would leave the job `running` with no audio; the backend
    reclaims it after its timeout, but that is 45 minutes of a chapter looking
    stuck. Draining is cheap and avoids it.
    """
    global _STOP
    log.info("signal %s received — finishing the current chapter, then stopping", signum)
    _STOP = True


def _post(api: str, path: str, token: str, payload: dict, timeout: int = 120) -> dict:
    import json

    req = urllib.request.Request(
        f"{api.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(prog="penspace.worker", description=__doc__)
    p.add_argument("--api", required=True, help="Backend API root, e.g. https://penspace.in/api")
    p.add_argument("--token", required=True, help="AUDIO_WORKER_TOKEN from the backend")
    p.add_argument("--work-dir", default="/workspace/out")
    p.add_argument("--bucket", help="Overrides PENSPACE_S3_BUCKET")
    p.add_argument("--batch", type=int, default=1, help="Chapters to claim per poll")
    p.add_argument("--idle-sleep", type=int, default=15, help="Seconds to wait when nothing is queued")
    p.add_argument("--once", action="store_true", help="Drain the queue and exit")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    cfg = Config.from_env()
    if args.bucket:
        cfg.s3_bucket = args.bucket
    if not cfg.s3_bucket:
        log.error("No bucket. Pass --bucket or set PENSPACE_S3_BUCKET, or the audio has nowhere to go.")
        return 1

    # Warn loudly rather than fail: an unpinned reference transcript is inferred
    # by Whisper, and a transcript that differs between runs changes the render
    # fingerprint — so identical text re-renders and is billed again.
    import os
    if cfg.is_clone and not os.environ.get("PENSPACE_REF_TEXT"):
        log.warning("PENSPACE_REF_TEXT is not pinned — renders may not dedupe against S3.")

    storage = S3Storage(cfg)
    try:
        storage.preflight()
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1

    log.info("loading model (once — every chapter after this reuses it)")
    cfg = prepare_clone(cfg)
    runner = Runner(cfg, Path(args.work_dir), storage)
    log.info("ready; polling %s", args.api)

    rendered_total = 0

    while not _STOP:
        try:
            claim = _post(args.api, "/admin/audio-worker/claim", args.token,
                          {"limit": args.batch, "workerId": f"pod-{os.uname().nodename}"})
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                log.error("worker token rejected — check AUDIO_WORKER_TOKEN on both sides")
                return 1
            log.warning("claim failed (%s); retrying", exc)
            time.sleep(args.idle_sleep)
            continue
        except Exception as exc:  # noqa: BLE001 — network flake, keep polling
            log.warning("claim failed (%s); retrying", exc)
            time.sleep(args.idle_sleep)
            continue

        jobs = (claim.get("data") or {}).get("jobs") or []
        if not jobs:
            if args.once:
                log.info("queue empty; rendered %s chapter(s) this run", rendered_total)
                return 0
            time.sleep(args.idle_sleep)
            continue

        rows, failed = [], []
        for job in jobs:
            started = time.time()
            try:
                result = runner.render(
                    SummaryJob(
                        id=job["id"],
                        text=job["text"],
                        title=job.get("title"),
                        author=job.get("author"),
                    )
                )
                row = result.to_dict()
                row["skipped"] = result.skipped
                rows.append(row)
                rendered_total += 1
                log.info(
                    "%s %s — %.1fs audio in %.0fs wall",
                    "skipped (already rendered)" if result.skipped else "rendered",
                    job["id"],
                    result.duration,
                    time.time() - started,
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("render failed for %s", job["id"])
                failed.append({"id": job["id"], "error": f"{type(exc).__name__}: {exc}"})

        # Reported even when everything failed, so the panel stops showing
        # "Rendering…" and says what went wrong instead.
        try:
            done = _post(args.api, "/admin/audio-worker/complete", args.token,
                         {"rows": rows, "failed": failed})
            log.info("reported: %s", (done.get("data") or {}))
        except Exception as exc:  # noqa: BLE001
            # The audio IS in S3; only the report failed. The backend reclaims
            # the job after its timeout and the re-render is a no-op, because
            # the fingerprint already matches an object in the bucket.
            log.error("could not report results (%s) — audio is in S3, job will be reclaimed", exc)

    log.info("stopped; rendered %s chapter(s) this run", rendered_total)
    return 0


if __name__ == "__main__":
    sys.exit(main())

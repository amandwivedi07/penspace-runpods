"""Command line entrypoint for the Penspace narration pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Config
from .runner import Runner, estimate, load_manifest
from .storage import S3Storage


# These emit megabytes of DEBUG at -v and drown out anything useful. numba in
# particular dumps its full SSA IR for every jitted librosa function.
NOISY_LOGGERS = (
    "numba", "urllib3", "botocore", "boto3", "s3transfer", "httpx", "httpcore",
    "filelock", "huggingface_hub", "matplotlib", "asyncio", "PIL",
)


def _configure_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _apply_overrides(cfg: Config, args) -> Config:
    for field in (
        "speaker", "language", "instruct", "device", "batch_size",
        "max_chunk_chars", "max_wer", "bitrate", "s3_prefix",
    ):
        value = getattr(args, field, None)
        if value is not None:
            setattr(cfg, field, value)
    if getattr(args, "bucket", None):
        cfg.s3_bucket = args.bucket
    if getattr(args, "no_qa", False):
        cfg.qa_enabled = False
    for field in ("ref_audio", "ref_text"):
        value = getattr(args, field, None)
        if value is not None:
            setattr(cfg, field, value)
    if getattr(args, "x_vector_only", False):
        cfg.x_vector_only = True
    # Supplying a reference recording is what selects clone mode.
    if cfg.ref_audio:
        cfg.voice_mode = "clone"
    return cfg


def _select(jobs, args):
    if args.only:
        wanted = set(args.only)
        jobs = [j for j in jobs if j.id in wanted]
    if args.limit:
        jobs = jobs[: args.limit]
    return jobs


def cmd_estimate(args) -> int:
    cfg = _apply_overrides(Config.from_env(), args)
    jobs = _select(load_manifest(Path(args.manifest)), args)
    report = estimate(jobs, cfg)
    print(json.dumps(report, indent=2))
    if report["oversized_chunks"]:
        print(
            f"\nWARNING: {report['oversized_chunks']} chunks exceed "
            f"{cfg.max_chunk_chars} chars and could not be split further.",
            file=sys.stderr,
        )
    return 0


def cmd_render(args) -> int:
    from .runner import prepare_clone

    cfg = prepare_clone(_apply_overrides(Config.from_env(), args))
    jobs = _select(load_manifest(Path(args.manifest)), args)
    if not jobs:
        print("nothing to render", file=sys.stderr)
        return 1

    storage = S3Storage(cfg) if cfg.s3_bucket else None
    if storage is None:
        logging.warning("no --bucket given; rendering locally without upload")

    work_dir = Path(args.work_dir)
    results = Runner(cfg, work_dir, storage).run(jobs)

    # The catalog is the handoff to the app: one row per playable summary.
    catalog = work_dir / "catalog.jsonl"
    with catalog.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r.to_dict()) + "\n")

    rendered = [r for r in results if not r.skipped]
    flagged = [r for r in results if r.failed_chunks]
    total_hours = sum(r.duration for r in results) / 3600.0

    print(f"\nrendered {len(rendered)}, skipped {len(results) - len(rendered)}, "
          f"failed {len(jobs) - len(results)}")
    print(f"total audio: {total_hours:.2f} h")
    print(f"catalog: {catalog}")
    if flagged:
        print(f"\n{len(flagged)} summaries have chunks that never passed QA:",
              file=sys.stderr)
        for r in flagged:
            print(f"  {r.id}: chunks {r.failed_chunks}", file=sys.stderr)
        return 2
    return 0


DEFAULT_AUDITION_TEXT = (
    "Atomic Habits argues that remarkable results come from small, consistent "
    "improvements rather than dramatic transformations. Clear introduces the "
    "idea of getting 1% better every day. Over a year, that compounds into a "
    "37x improvement.\n\n"
    "He frames habit change around four laws. Make it obvious, make it "
    "attractive, make it easy, and make it satisfying."
)


def cmd_audition(args) -> int:
    from .runner import audition, prepare_clone

    cfg = prepare_clone(_apply_overrides(Config.from_env(), args))
    text = Path(args.text_file).read_text() if args.text_file else args.text
    # In clone mode there is only one voice: the reference.
    speakers = ["clone"] if cfg.is_clone else args.speakers
    results = audition(cfg, text, speakers, Path(args.out))

    print()
    for r in results:
        print(
            f"{r['speaker']:<10} {r['audio_seconds']:>7.2f}s audio  "
            f"{r['generation_seconds']:>7.2f}s generate  "
            f"{r['realtime_factor']:>6.2f}x realtime  {r['path']}"
        )
    return 0


def cmd_ingest(args) -> int:
    from .ingest import ingest

    result = ingest(
        source=Path(args.source),
        out_dir=Path(args.out_dir),
        summary_id=args.id,
        title=args.title,
        author=args.author,
        manifest=Path(args.manifest) if args.manifest else None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_url(args) -> int:
    cfg = _apply_overrides(Config.from_env(), args)
    if not cfg.s3_bucket:
        print("--bucket or PENSPACE_S3_BUCKET is required", file=sys.stderr)
        return 1
    print(S3Storage(cfg).presign(args.key, args.ttl))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="penspace", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--speaker")
        sp.add_argument("--language")
        sp.add_argument("--instruct")
        sp.add_argument("--device")
        sp.add_argument("--batch-size", type=int, dest="batch_size")
        sp.add_argument("--max-chunk-chars", type=int, dest="max_chunk_chars")
        sp.add_argument("--bitrate")
        sp.add_argument("--only", nargs="*", help="render only these summary ids")
        sp.add_argument("--limit", type=int, help="render at most N summaries")
        sp.add_argument(
            "--ref-audio",
            dest="ref_audio",
            help="reference recording to clone; switches to the Base model",
        )
        sp.add_argument(
            "--ref-text",
            dest="ref_text",
            help="transcript of --ref-audio (auto-transcribed if omitted)",
        )
        sp.add_argument(
            "--x-vector-only",
            dest="x_vector_only",
            action="store_true",
            help="clone from the speaker embedding alone; no transcript, lower fidelity",
        )

    est = sub.add_parser("estimate", help="chunk the manifest without using a GPU")
    est.add_argument("--manifest", required=True)
    add_common(est)
    est.set_defaults(func=cmd_estimate)

    ren = sub.add_parser("render", help="synthesize, QA, master and upload")
    ren.add_argument("--manifest", required=True)
    ren.add_argument("--work-dir", default="./penspace_out")
    ren.add_argument("--bucket")
    ren.add_argument("--s3-prefix", dest="s3_prefix")
    ren.add_argument("--max-wer", type=float, dest="max_wer")
    ren.add_argument("--no-qa", action="store_true", help="skip the ASR quality gate")
    add_common(ren)
    ren.set_defaults(func=cmd_render)

    ing = sub.add_parser("ingest", help="convert a .docx/.md/.txt into a manifest entry")
    ing.add_argument("source", help="path to the source document")
    ing.add_argument("--out-dir", dest="out_dir", default="./summaries")
    ing.add_argument("--id", help="summary id (defaults to a slug of the filename)")
    ing.add_argument("--title")
    ing.add_argument("--author")
    ing.add_argument("--manifest", help="manifest to create or update")
    ing.set_defaults(func=cmd_ingest)

    aud = sub.add_parser("audition", help="render one passage in several voices")
    aud.add_argument("--text", default=DEFAULT_AUDITION_TEXT)
    aud.add_argument("--text-file", dest="text_file")
    aud.add_argument("--speakers", nargs="+", default=["Ryan", "Aiden"])
    aud.add_argument("--out", default="./penspace_audition")
    add_common(aud)
    aud.set_defaults(func=cmd_audition)

    url = sub.add_parser("url", help="mint a presigned playback URL")
    url.add_argument("--key", required=True)
    url.add_argument("--bucket")
    url.add_argument("--ttl", type=int, default=3600)
    add_common(url)
    url.set_defaults(func=cmd_url)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

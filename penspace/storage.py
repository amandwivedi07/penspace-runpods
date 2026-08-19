"""S3 upload and presigned playback URLs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .config import Config

log = logging.getLogger(__name__)

CONTENT_TYPES = {
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".json": "application/json",
}


def render_key(prefix: str, summary_id: str, render_id: str, filename: str) -> str:
    """Content-addressed key.

    `render_id` is a hash of the text plus voice settings, so any re-render
    lands on a new key. That is what makes `immutable` cache headers safe and
    keeps CDN egress near zero for repeat plays.
    """
    return f"{prefix.rstrip('/')}/{summary_id}/{render_id}/{filename}"


class S3Storage:
    def __init__(self, cfg: Config):
        if not cfg.s3_bucket:
            raise ValueError("s3_bucket is required for S3Storage")
        import boto3

        self.cfg = cfg
        self.bucket = cfg.s3_bucket
        self.client = boto3.client("s3", region_name=cfg.s3_region)

    def preflight(self) -> None:
        """Prove we can write to the bucket before spending GPU time.

        A missing credential only surfaces at the first upload, which is after
        a full chapter has been synthesised, WER-checked and mastered — around
        90 seconds of GPU per chapter, silently wasted on every job in the
        queue. Failing here instead costs one API call.
        """
        from botocore.exceptions import ClientError, NoCredentialsError

        try:
            # head_bucket has to sign the request, so a missing or unusable
            # credential surfaces here as NoCredentialsError.
            self.client.head_bucket(Bucket=self.bucket)
        except NoCredentialsError:
            raise RuntimeError(
                "No AWS credentials. Export AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY (and AWS_DEFAULT_REGION) before starting "
                "the worker — audio would render fine and then fail to upload."
            ) from None
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("404", "NoSuchBucket"):
                raise RuntimeError(
                    f"Bucket {self.bucket!r} does not exist in region "
                    f"{self.cfg.s3_region!r}. Check --bucket and AWS_DEFAULT_REGION."
                ) from None
            if code in ("403", "AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"):
                raise RuntimeError(
                    f"Credentials rejected for bucket {self.bucket!r} ({code}). "
                    "Either the key/secret is wrong, or it lacks s3:PutObject "
                    "and s3:ListBucket on that bucket."
                ) from None
            raise RuntimeError(
                f"S3 preflight failed for {self.bucket!r}: {code}"
            ) from None
        log.info("s3 preflight ok: s3://%s (%s)", self.bucket, self.cfg.s3_region)

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "403"):
                return False
            raise

    def upload(self, path: Path, key: str) -> str:
        content_type = CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        extra = {"ContentType": content_type, "CacheControl": self.cfg.cache_control}
        log.info("uploading %s -> s3://%s/%s", path.name, self.bucket, key)
        self.client.upload_file(str(path), self.bucket, key, ExtraArgs=extra)
        return f"s3://{self.bucket}/{key}"

    def presign(self, key: str, ttl: Optional[int] = None) -> str:
        """Time-limited playback URL.

        Supports HTTP range requests, so mobile players can seek without
        downloading the whole file.
        """
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=ttl or self.cfg.presign_ttl,
        )

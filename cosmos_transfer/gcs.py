"""Google Cloud Storage upload, signed URLs, and download."""

from __future__ import annotations

import os
import uuid
from datetime import timedelta
from pathlib import Path

DEFAULT_PREFIX = "cosmos-transfer"
DEFAULT_SIGNED_URL_MINUTES = 120


def _bucket_name() -> str:
    name = os.environ.get("GCS_BUCKET", "").strip()
    if not name:
        raise RuntimeError("GCS_BUCKET is not set")
    return name


def _prefix() -> str:
    return os.environ.get("GCS_PREFIX", DEFAULT_PREFIX).strip().strip("/")


def _client():
    from google.cloud import storage

    return storage.Client()


def object_name(kind: str, job_id: str, filename: str) -> str:
    """Build GCS object key: {prefix}/{kind}/{job_id}/{filename}."""
    return f"{_prefix()}/{kind}/{job_id}/{filename}"


def upload_file(local_path: Path, *, kind: str = "inputs", job_id: str | None = None) -> str:
    """Upload a file; return ``gs://bucket/object`` URI."""
    local_path = Path(local_path).resolve()
    job_id = job_id or uuid.uuid4().hex
    blob_name = object_name(kind, job_id, local_path.name)
    bucket = _client().bucket(_bucket_name())
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    return f"gs://{_bucket_name()}/{blob_name}"


def signed_url(
    gs_uri: str,
    *,
    expiration_minutes: int = DEFAULT_SIGNED_URL_MINUTES,
    method: str = "GET",
) -> str:
    """Generate a V4 signed URL for a ``gs://`` URI."""
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {gs_uri}")
    rest = gs_uri[5:]
    bucket_name, _, blob_name = rest.partition("/")
    bucket = _client().bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=expiration_minutes),
        method=method,
    )


def download_url(url: str, dest_path: Path) -> Path:
    """Download ``url`` (https or gs) to ``dest_path``."""
    import urllib.request

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if url.startswith("gs://"):
        url = signed_url(url, method="GET")
    urllib.request.urlretrieve(url, dest_path)
    return dest_path


def download_gs_uri(gs_uri: str, dest_path: Path) -> Path:
    """Download a ``gs://`` object via the storage client."""
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {gs_uri}")
    rest = gs_uri[5:]
    bucket_name, _, blob_name = rest.partition("/")
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    bucket = _client().bucket(bucket_name)
    bucket.blob(blob_name).download_to_filename(str(dest_path))
    return dest_path

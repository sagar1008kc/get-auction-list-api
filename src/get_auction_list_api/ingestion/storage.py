"""Server-credential download for ``supabase://`` Storage URIs.

Never invent browser signed URLs; callers must provide the project URL and service role.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

import httpx


class StorageDownloadError(OSError):
    """Raised when Storage refuses or cannot return object bytes."""


@dataclass(frozen=True, slots=True)
class StorageObject:
    bucket: str
    path: str
    content: bytes
    media_type: str | None


def parse_supabase_uri(source_uri: str) -> tuple[str, str]:
    if not source_uri.startswith("supabase://"):
        raise ValueError("Source URI must use the supabase:// scheme.")
    remainder = source_uri.removeprefix("supabase://")
    bucket, separator, path = remainder.partition("/")
    if not separator or not bucket or not path:
        raise ValueError("supabase:// URI must be supabase://{bucket}/{object_path}.")
    if ".." in path.split("/") or path.startswith("/"):
        raise ValueError("Storage object path is invalid.")
    return bucket, path


async def download_supabase_object(
    *,
    supabase_url: str,
    service_role_key: str,
    bucket: str,
    path: str,
    timeout_seconds: float = 30,
) -> StorageObject:
    """Download an object with the service role; never return a signed browser URL."""

    base = supabase_url.rstrip("/")
    encoded_path = "/".join(quote(part, safe="") for part in path.split("/") if part)
    url = f"{base}/storage/v1/object/{quote(bucket, safe='')}/{encoded_path}"
    headers = {
        "Authorization": f"Bearer {service_role_key}",
        "apikey": service_role_key,
    }
    async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=False) as client:
        response = await client.get(url, headers=headers)
    if response.status_code == 404:
        raise StorageDownloadError(f"Storage object not found: {bucket}/{path}")
    if response.status_code >= 400:
        raise StorageDownloadError(
            f"Storage download failed with HTTP {response.status_code} for {bucket}/{path}."
        )
    media_type = response.headers.get("content-type")
    return StorageObject(
        bucket=bucket,
        path=path,
        content=response.content,
        media_type=media_type.split(";", 1)[0].strip() if media_type else None,
    )

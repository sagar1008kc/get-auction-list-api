"""Bounded source-byte validation before parser dispatch."""

import hashlib
import io
import zipfile
from dataclasses import dataclass


class FileValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ValidatedFile:
    content: bytes
    media_type: str
    sha256: str
    byte_size: int


_SIGNATURES = {
    "application/pdf": (b"%PDF-",),
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": (b"PK\x03\x04",),
}
_HTML_PREFIXES = (b"<!doctype html", b"<html", b"<head", b"<body")


def _detected_media_type(content: bytes) -> str:
    prefix = content[:512].lstrip().lower()
    if content.startswith(b"%PDF-"):
        return "application/pdf"
    if content.startswith(b"PK\x03\x04"):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if prefix.startswith(_HTML_PREFIXES):
        return "text/html"
    raise FileValidationError("Unsupported or unrecognized file signature.")


def _validate_zip(content: bytes, *, max_uncompressed_bytes: int, max_ratio: float) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            infos = archive.infolist()
            if len(infos) > 10_000:
                raise FileValidationError("Archive contains too many entries.")
            expanded = sum(info.file_size for info in infos)
            compressed = sum(max(info.compress_size, 1) for info in infos)
            if expanded > max_uncompressed_bytes or expanded / compressed > max_ratio:
                raise FileValidationError("Archive expansion exceeds safety limits.")
            if not any(info.filename == "[Content_Types].xml" for info in infos):
                raise FileValidationError("ZIP is not an XLSX package.")
    except zipfile.BadZipFile as error:
        raise FileValidationError("Malformed XLSX archive.") from error


def validate_file(
    content: bytes,
    *,
    declared_media_type: str | None,
    max_bytes: int = 10_000_000,
    max_uncompressed_bytes: int = 100_000_000,
    max_zip_ratio: float = 100.0,
) -> ValidatedFile:
    if not content:
        raise FileValidationError("File is empty.")
    if len(content) > max_bytes:
        raise FileValidationError("File exceeds the configured byte limit.")
    detected = _detected_media_type(content)
    normalized_declared = (
        declared_media_type.split(";", 1)[0].strip().lower() if declared_media_type else None
    )
    if normalized_declared and normalized_declared != detected:
        raise FileValidationError("Declared MIME type does not match the file signature.")
    if detected in _SIGNATURES and not content.startswith(_SIGNATURES[detected]):
        raise FileValidationError("File signature does not match its detected MIME type.")
    if detected.endswith("spreadsheetml.sheet"):
        _validate_zip(
            content,
            max_uncompressed_bytes=max_uncompressed_bytes,
            max_ratio=max_zip_ratio,
        )
    return ValidatedFile(
        content=content,
        media_type=detected,
        sha256=hashlib.sha256(content).hexdigest(),
        byte_size=len(content),
    )

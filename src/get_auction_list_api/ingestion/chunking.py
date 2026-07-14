"""Heading-aware deterministic chunking with coordinate preservation."""

import hashlib

from pydantic import BaseModel, ConfigDict, Field

from get_auction_list_api.parsers.models import ParsedUnit, SourceCoordinates


class DocumentChunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ordinal: int = Field(ge=0)
    content: str
    content_sha256: str
    token_count: int = Field(ge=0)
    coordinates: SourceCoordinates


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def chunk_units(
    units: tuple[ParsedUnit, ...],
    *,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> tuple[DocumentChunk, ...]:
    if max_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("Chunk budget is invalid.")
    chunks: list[DocumentChunk] = []
    for unit in units:
        heading = " > ".join(unit.coordinates.section_path)
        prefix = f"{heading}\n" if heading else ""
        words = unit.text.split()
        words_per_chunk = max(1, max_tokens * 4 // 6)
        overlap_words = overlap_tokens * 4 // 6
        cursor = 0
        while cursor < len(words):
            body = " ".join(words[cursor : cursor + words_per_chunk])
            content = (prefix + body).strip()
            chunks.append(
                DocumentChunk(
                    ordinal=len(chunks),
                    content=content,
                    content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                    token_count=_estimate_tokens(content),
                    coordinates=unit.coordinates,
                )
            )
            if cursor + words_per_chunk >= len(words):
                break
            cursor += max(1, words_per_chunk - overlap_words)
    return tuple(chunks)

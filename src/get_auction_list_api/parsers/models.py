"""Parser output contracts preserve immutable source coordinates and row failures."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SourceCoordinates(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    page_number: int | None = Field(default=None, ge=1)
    sheet_name: str | None = None
    row_start: int | None = Field(default=None, ge=1)
    row_end: int | None = Field(default=None, ge=1)
    section_path: tuple[str, ...] = ()
    url: str | None = None


class ParsedUnit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    coordinates: SourceCoordinates
    fields: dict[str, Any] = Field(default_factory=dict)


class RowError(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    message: str
    coordinates: SourceCoordinates


class ParseResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    units: tuple[ParsedUnit, ...]
    errors: tuple[RowError, ...] = ()

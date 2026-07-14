"""Read-only openpyxl parser with row-level rejection."""

import io
import zipfile
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import openpyxl

from get_auction_list_api.parsers.models import (
    ParsedUnit,
    ParseResult,
    RowError,
    SourceCoordinates,
)


def _cell_value(value: object) -> Any:
    if value is None or isinstance(value, (str, bool, int, date, datetime, Decimal)):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    return str(value)


class XlsxParser:
    name = "openpyxl"
    version = openpyxl.__version__

    def parse(self, content: bytes) -> ParseResult:
        units: list[ParsedUnit] = []
        errors: list[RowError] = []
        try:
            workbook = openpyxl.load_workbook(
                io.BytesIO(content),
                read_only=True,
                data_only=True,
                keep_links=False,
            )
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            raise ValueError("Malformed XLSX workbook.") from error
        try:
            for sheet in workbook.worksheets:
                rows = sheet.iter_rows(values_only=True)
                raw_headers = next(rows, None)
                if raw_headers is None:
                    continue
                headers = [str(value).strip() if value is not None else "" for value in raw_headers]
                if not any(headers) or len(set(header for header in headers if header)) != sum(
                    bool(header) for header in headers
                ):
                    errors.append(
                        RowError(
                            code="invalid_header",
                            message="Header row is empty or contains duplicate names.",
                            coordinates=SourceCoordinates(
                                sheet_name=sheet.title, row_start=1, row_end=1
                            ),
                        )
                    )
                    continue
                for row_number, values in enumerate(rows, start=2):
                    coordinates = SourceCoordinates(
                        sheet_name=sheet.title,
                        row_start=row_number,
                        row_end=row_number,
                    )
                    if not any(value is not None and str(value).strip() for value in values):
                        continue
                    try:
                        fields = {
                            header: _cell_value(value)
                            for header, value in zip(headers, values, strict=False)
                            if header
                        }
                        text = "\n".join(
                            f"{key}: {value}" for key, value in fields.items() if value is not None
                        )
                        units.append(ParsedUnit(text=text, coordinates=coordinates, fields=fields))
                    except (TypeError, ValueError) as error:
                        errors.append(
                            RowError(
                                code="invalid_row",
                                message=str(error)[:300],
                                coordinates=coordinates,
                            )
                        )
        finally:
            workbook.close()
        return ParseResult(units=tuple(units), errors=tuple(errors))

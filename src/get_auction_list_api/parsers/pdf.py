"""PyMuPDF parser producing page-addressable evidence."""

import pymupdf

from get_auction_list_api.parsers.models import ParsedUnit, ParseResult, SourceCoordinates


class PdfParser:
    name = "pymupdf"
    version = pymupdf.__version__

    def parse(self, content: bytes) -> ParseResult:
        units: list[ParsedUnit] = []
        try:
            with pymupdf.open(stream=content, filetype="pdf") as document:  # type: ignore[no-untyped-call]
                if document.needs_pass:
                    raise ValueError("Encrypted PDFs are not supported.")
                for number, page in enumerate(document, start=1):
                    text = page.get_text("text", sort=True).strip()
                    if text:
                        units.append(
                            ParsedUnit(
                                text=text,
                                coordinates=SourceCoordinates(page_number=number),
                            )
                        )
        except pymupdf.FileDataError as error:
            raise ValueError("Malformed PDF.") from error
        return ParseResult(units=tuple(units))

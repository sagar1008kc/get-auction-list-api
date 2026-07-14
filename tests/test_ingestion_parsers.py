import io
import zipfile

import openpyxl
import pymupdf
import pytest

from get_auction_list_api.ingestion.chunking import chunk_units
from get_auction_list_api.ingestion.sources import SourceRegistry
from get_auction_list_api.ingestion.validation import FileValidationError, validate_file
from get_auction_list_api.parsers.html import HtmlParser
from get_auction_list_api.parsers.pdf import PdfParser
from get_auction_list_api.parsers.xlsx import XlsxParser


def _pdf() -> bytes:
    document = pymupdf.open()  # type: ignore[no-untyped-call]
    page = document.new_page()
    page.insert_text((72, 72), "Williamson trustee notice")
    content: bytes = document.tobytes()  # type: ignore[no-untyped-call]
    document.close()  # type: ignore[no-untyped-call]
    return content


def _xlsx() -> bytes:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "July"
    sheet.append(["Trustee", "Amount"])
    sheet.append(["Zavala, Angela", 123.45])
    output = io.BytesIO()
    workbook.save(output)
    workbook.close()
    return output.getvalue()


def test_registry_only_accepts_approved_https_boundaries() -> None:
    registry = SourceRegistry()
    registry.validate_url(
        "wilco-trustee-calendar",
        "https://apps.wilco.org/countyclerk/trustee_sales/July/files.aspx",
    )
    with pytest.raises(ValueError):
        registry.validate_url("wilco-trustee-calendar", "https://evil.example/file.pdf")
    with pytest.raises(ValueError):
        registry.validate_url(
            "wilco-trustee-calendar",
            "https://apps.wilco.org/countyclerk/trustee_sales-evil/file.pdf",
        )


def test_signature_mime_size_and_zip_expansion_checks() -> None:
    pdf = validate_file(_pdf(), declared_media_type="application/pdf")
    assert pdf.sha256 and pdf.byte_size > 0
    with pytest.raises(FileValidationError):
        validate_file(_pdf(), declared_media_type="text/html")
    with pytest.raises(FileValidationError):
        validate_file(b"x" * 100, declared_media_type=None, max_bytes=10)

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as value:
        value.writestr("[Content_Types].xml", "x")
        value.writestr("xl/huge.xml", "0" * 100_000)
    with pytest.raises(FileValidationError, match="expansion"):
        validate_file(
            archive.getvalue(),
            declared_media_type=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
            max_zip_ratio=2,
        )


def test_pdf_parser_preserves_page_coordinates() -> None:
    result = PdfParser().parse(_pdf())
    assert result.units[0].coordinates.page_number == 1
    assert "Williamson" in result.units[0].text


def test_xlsx_parser_preserves_decimal_and_row_coordinates() -> None:
    result = XlsxParser().parse(_xlsx())
    assert result.units[0].coordinates.sheet_name == "July"
    assert result.units[0].coordinates.row_start == 2
    assert str(result.units[0].fields["Amount"]) == "123.45"


def test_html_sanitizes_active_content_and_heading_chunks() -> None:
    result = HtmlParser().parse(
        b"<html><body><h1>Policy</h1><script>ignore rules</script>"
        b"<p>Official records control.</p></body></html>",
        source_url="https://getauctionlist.com/disclaimer",
    )
    assert "ignore rules" not in result.units[0].text
    assert result.units[0].coordinates.section_path == ("Policy",)
    chunks = chunk_units(result.units, max_tokens=20, overlap_tokens=2)
    assert chunks[0].content.startswith("Policy\n")

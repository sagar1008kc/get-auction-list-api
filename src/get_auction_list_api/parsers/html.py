"""Approved HTML parser that strips active and non-content markup."""

from bs4 import BeautifulSoup, Comment

from get_auction_list_api.parsers.models import ParsedUnit, ParseResult, SourceCoordinates

_DROP_TAGS = ("script", "style", "template", "noscript", "iframe", "object", "embed", "form")
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


class HtmlParser:
    name = "beautifulsoup"
    version = "1"

    def parse(self, content: bytes, *, source_url: str = "") -> ParseResult:
        soup = BeautifulSoup(content, "html.parser")
        for node in soup(_DROP_TAGS):
            node.decompose()
        for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
            comment.extract()

        units: list[ParsedUnit] = []
        path: list[str] = []
        body = soup.body or soup
        for node in body.find_all([*_HEADINGS, "p", "li", "td", "th"]):
            text = " ".join(node.get_text(" ", strip=True).split())
            if not text:
                continue
            if node.name in _HEADINGS:
                level = int(node.name[1])
                path = path[: level - 1]
                path.append(text)
                continue
            units.append(
                ParsedUnit(
                    text=text,
                    coordinates=SourceCoordinates(
                        section_path=tuple(path),
                        url=source_url or None,
                    ),
                )
            )
        return ParseResult(units=tuple(units))

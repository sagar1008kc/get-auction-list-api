"""Immutable registry of sources approved by the architecture review."""

from enum import StrEnum
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, HttpUrl


class SourceKind(StrEnum):
    POLICY_HTML = "policy_html"
    PUBLIC_HTML = "public_html"
    PUBLIC_PDF = "public_pdf"


class ApprovedSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    title: str
    base_url: HttpUrl
    kind: SourceKind
    official: bool

    def permits(self, candidate: str) -> bool:
        approved = urlsplit(str(self.base_url))
        value = urlsplit(candidate)
        return (
            value.scheme == "https"
            and value.hostname == approved.hostname
            and value.port in (None, 443)
            and value.username is None
            and value.password is None
            and (
                value.path == approved.path
                or value.path.startswith(approved.path.rstrip("/") + "/")
            )
        )


APPROVED_SOURCES: tuple[ApprovedSource, ...] = (
    ApprovedSource(
        key="getauctionlist-privacy",
        title="GetAuctionList Privacy Policy",
        base_url=HttpUrl("https://getauctionlist.com/privacy"),
        kind=SourceKind.POLICY_HTML,
        official=False,
    ),
    ApprovedSource(
        key="getauctionlist-disclaimer",
        title="GetAuctionList Disclaimer",
        base_url=HttpUrl("https://getauctionlist.com/disclaimer"),
        kind=SourceKind.POLICY_HTML,
        official=False,
    ),
    ApprovedSource(
        key="wilco-trustee-calendar",
        title="Williamson County Clerk Trustee Sales",
        base_url=HttpUrl("https://apps.wilco.org/countyclerk/trustee_sales/"),
        kind=SourceKind.PUBLIC_HTML,
        official=True,
    ),
    ApprovedSource(
        key="wilco-foreclosure-sales",
        title="Williamson County Foreclosure Trustee Sales",
        base_url=HttpUrl("https://www.wilcotx.gov/308/Foreclosure-Trustee-Sales"),
        kind=SourceKind.PUBLIC_HTML,
        official=True,
    ),
)


class SourceRegistry:
    def __init__(self, sources: tuple[ApprovedSource, ...] = APPROVED_SOURCES) -> None:
        self._sources = {source.key: source for source in sources}

    def get(self, key: str) -> ApprovedSource:
        try:
            return self._sources[key]
        except KeyError as error:
            raise ValueError("Source is not registered or approved.") from error

    def validate_url(self, key: str, url: str) -> ApprovedSource:
        source = self.get(key)
        if not source.permits(url):
            raise ValueError("URL is outside the approved source boundary.")
        return source

    def all(self) -> tuple[ApprovedSource, ...]:
        return tuple(self._sources.values())

"""One-shot bootstrap: ingest July 2026 spreadsheet + policy HTML pages."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from get_auction_list_api.ingestion.handler import handle_ingestion_job
from get_auction_list_api.ingestion.queue import IngestionJob


def _job(source_uri: str, payload: dict[str, object]) -> IngestionJob:
    return IngestionJob(
        id=uuid4(),
        idempotency_key=f"bootstrap:{source_uri}",
        source_uri=source_uri,
        state="running",
        attempt_count=1,
        input=payload,
    )


async def main() -> None:
    auction_path = "williamson_county/getAuctionList_July_2026.xlsx"
    # Prefer capital-J path; handler will 404 clearly if Storage uses another casing.
    auction = await handle_ingestion_job(
        _job(
            f"supabase://auction_files/{auction_path}",
            {
                "source_type": "supabase_storage",
                "storage_bucket": "auction_files",
                "storage_path": auction_path,
                "document_type": "auction_spreadsheet",
            },
        )
    )
    print("auction_ingest", {k: auction[k] for k in auction if k != "document_id"})

    for key in ("getauctionlist-privacy", "getauctionlist-disclaimer"):
        result = await handle_ingestion_job(
            _job(
                f"registered://{key}",
                {
                    "source_type": "registered_url",
                    "source_id": key,
                    "document_type": "policy_html",
                },
            )
        )
        print("policy_ingest", key, {k: result[k] for k in result if k != "document_id"})


if __name__ == "__main__":
    asyncio.run(main())

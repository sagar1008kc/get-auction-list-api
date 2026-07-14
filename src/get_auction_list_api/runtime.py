"""Small dependency-injected defaults for database/model-free local execution."""

import asyncio
from uuid import UUID, uuid4

from get_auction_list_api.auth import Principal
from get_auction_list_api.schemas import FeedbackRequest


class InMemoryFeedbackStore:
    def __init__(self) -> None:
        self._values: dict[tuple[UUID, UUID], UUID] = {}
        self._lock = asyncio.Lock()

    async def record(
        self,
        principal: Principal,
        feedback: FeedbackRequest,
    ) -> tuple[UUID, bool]:
        key = (principal.user_id, feedback.message_id)
        async with self._lock:
            existing = self._values.get(key)
            if existing is not None:
                return existing, False
            feedback_id = uuid4()
            self._values[key] = feedback_id
            return feedback_id, True

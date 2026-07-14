"""Provider-neutral embedding contracts and an injectable OpenAI transport."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from get_auction_list_api.api.metrics import record_model_usage
from get_auction_list_api.observability.telemetry import tracer


class EmbeddingBatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    dimensions: int = Field(gt=0)
    vectors: tuple[tuple[float, ...], ...]


class EmbeddingProvider(Protocol):
    async def embed(self, texts: Sequence[str]) -> EmbeddingBatch: ...


OpenAIEmbeddingTransport = Callable[
    [str, Sequence[str], int],
    Awaitable[Sequence[Sequence[float]]],
]


class OpenAIEmbeddingProvider:
    """OpenAI adapter keeps credentials and HTTP concerns in the injected transport."""

    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        transport: OpenAIEmbeddingTransport,
        max_batch_size: int = 100,
    ) -> None:
        if not model or dimensions <= 0 or max_batch_size <= 0:
            raise ValueError("Valid embedding model, dimensions, and batch size are required.")
        self._model = model
        self._dimensions = dimensions
        self._transport = transport
        self._max_batch_size = max_batch_size

    async def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        clean = tuple(text.strip() for text in texts)
        if not clean or any(not text for text in clean):
            raise ValueError("Embedding input must contain non-empty text.")
        vectors: list[tuple[float, ...]] = []
        with tracer().start_as_current_span("model.embedding") as span:
            span.set_attribute("gen_ai.operation.name", "embeddings")
            span.set_attribute("gen_ai.request.model", self._model)
            for start in range(0, len(clean), self._max_batch_size):
                response = await self._transport(
                    self._model,
                    clean[start : start + self._max_batch_size],
                    self._dimensions,
                )
                vectors.extend(tuple(float(value) for value in vector) for vector in response)
        if len(vectors) != len(clean) or any(len(vector) != self._dimensions for vector in vectors):
            raise ValueError("Embedding provider returned an invalid shape.")
        record_model_usage("embedding", input_tokens=sum(len(text.split()) for text in clean))
        return EmbeddingBatch(
            model=self._model,
            dimensions=self._dimensions,
            vectors=tuple(vectors),
        )

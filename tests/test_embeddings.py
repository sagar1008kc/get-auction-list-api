from collections.abc import Sequence

import pytest

from get_auction_list_api.llm.embeddings import OpenAIEmbeddingProvider


@pytest.mark.asyncio
async def test_openai_embedding_adapter_batches_and_validates_shape() -> None:
    calls: list[tuple[str, Sequence[str], int]] = []

    async def transport(
        model: str, texts: Sequence[str], dimensions: int
    ) -> Sequence[Sequence[float]]:
        calls.append((model, texts, dimensions))
        return [(1.0, 2.0) for _ in texts]

    provider = OpenAIEmbeddingProvider(
        model="text-embedding-test",
        dimensions=2,
        transport=transport,
        max_batch_size=2,
    )
    batch = await provider.embed(("one", "two", "three"))
    assert len(calls) == 2
    assert batch.vectors[-1] == (1.0, 2.0)


@pytest.mark.asyncio
async def test_openai_embedding_adapter_rejects_wrong_dimensions() -> None:
    async def transport(
        model: str, texts: Sequence[str], dimensions: int
    ) -> Sequence[Sequence[float]]:
        return [(1.0,) for _ in texts]

    provider = OpenAIEmbeddingProvider(
        model="text-embedding-test",
        dimensions=2,
        transport=transport,
    )
    with pytest.raises(ValueError, match="shape"):
        await provider.embed(("one",))

"""Injectable OpenAI transports for embeddings and structured generation."""

import json
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar, cast

from openai import APIError, AsyncOpenAI
from pydantic import BaseModel

from get_auction_list_api.llm.embeddings import OpenAIEmbeddingTransport

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class OpenAIStructuredModel:
    """Validate structured model output at the application boundary."""

    def __init__(self, *, model: str, client: AsyncOpenAI) -> None:
        if not model:
            raise ValueError("An OpenAI chat model is required.")
        self._model = model
        self._client = client

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_type: type[ResponseT],
    ) -> ResponseT:
        schema = response_type.model_json_schema()
        try:
            response = await self._client.responses.create(
                model=self._model,
                instructions=system_prompt,
                input=user_prompt,
                text=cast(
                    Any,
                    {
                        "format": {
                            "type": "json_schema",
                            "name": response_type.__name__,
                            "schema": schema,
                            "strict": True,
                        }
                    },
                ),
            )
        except APIError as error:
            raise OSError("OpenAI structured generation failed.") from error
        if not response.output_text:
            raise ValueError("OpenAI returned an empty structured response.")
        payload = json.loads(response.output_text)
        if not isinstance(payload, Mapping):
            raise ValueError("OpenAI structured response must be an object.")
        return response_type.model_validate(payload)


def create_openai_client(
    *,
    api_key: str,
    base_url: str | None,
    timeout_seconds: float,
    max_retries: int,
) -> AsyncOpenAI:
    """Create one process-scoped OpenAI client."""

    if not api_key:
        raise ValueError("An OpenAI API key is required.")
    kwargs: dict[str, object] = {
        "api_key": api_key,
        "timeout": timeout_seconds,
        "max_retries": max_retries,
    }
    if base_url is not None:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**cast(Any, kwargs))


def create_embedding_transport(client: AsyncOpenAI) -> OpenAIEmbeddingTransport:
    """Adapt the OpenAI SDK to the provider-neutral embedding contract."""

    async def transport(
        model: str,
        texts: Sequence[str],
        dimensions: int,
    ) -> Sequence[Sequence[float]]:
        try:
            response = await client.embeddings.create(
                model=model,
                input=list(texts),
                dimensions=dimensions,
                encoding_format="float",
            )
        except APIError as error:
            raise OSError("OpenAI embedding generation failed.") from error
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    return transport

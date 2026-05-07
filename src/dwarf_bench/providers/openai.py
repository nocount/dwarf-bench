"""OpenAI provider. Chat Completions API; no tools; offline-only by construction."""
from __future__ import annotations

from typing import Any

from dwarf_bench.providers.base import ModelResponse


class OpenAIProvider:
    name = "openai"

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "OpenAIProvider requires the 'openai' SDK. "
                    "Install it with: pip install -e '.[openai]'"
                ) from e
            client = AsyncOpenAI()
        self._client = client

    async def generate(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        # Chat Completions: system + user messages, no tools.
        # `max_completion_tokens` is the modern parameter name and works for
        # both classic chat models (gpt-4o) and reasoning models (o3, gpt-5).
        resp = await self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        return ModelResponse(
            text=text,
            model=resp.model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            raw=resp.model_dump(mode="json"),
        )

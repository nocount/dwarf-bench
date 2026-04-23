"""Anthropic provider. Enforces offline-only generation by never passing tools."""
from __future__ import annotations

from anthropic import AsyncAnthropic

from dwarf_bench.providers.base import ModelResponse


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, client: AsyncAnthropic | None = None) -> None:
        self._client = client or AsyncAnthropic()

    async def generate(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        resp = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text_parts = [block.text for block in resp.content if block.type == "text"]
        text = "".join(text_parts).strip()
        return ModelResponse(
            text=text,
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            raw=resp.model_dump(mode="json"),
        )

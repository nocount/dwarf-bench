"""Provider abstraction. Any LLM backend implements the Provider protocol."""
from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel


class ModelResponse(BaseModel):
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    raw: dict[str, Any]


class Provider(Protocol):
    name: str

    async def generate(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        """Generate a response. Must not enable any tools / retrieval features."""
        ...

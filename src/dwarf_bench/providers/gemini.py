"""Google Gemini provider via the google-genai SDK. No grounding/tools."""
from __future__ import annotations

import os
from typing import Any

from dwarf_bench.providers.base import ModelResponse


class GeminiProvider:
    name = "google"

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            try:
                from google import genai
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "GeminiProvider requires the 'google-genai' SDK. "
                    "Install it with: pip install -e '.[gemini]'"
                ) from e
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self._client = client

    async def generate(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        # Pass system instruction in config; do NOT pass tools (no grounding,
        # no Google Search, no code execution). google-genai exposes async
        # via client.aio.
        from google.genai import types  # local import; SDK already loaded

        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        )
        resp = await self._client.aio.models.generate_content(
            model=model,
            contents=user,
            config=config,
        )
        text = (resp.text or "").strip()
        usage = getattr(resp, "usage_metadata", None)
        return ModelResponse(
            text=text,
            model=model,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            raw=resp.model_dump(mode="json") if hasattr(resp, "model_dump") else {},
        )

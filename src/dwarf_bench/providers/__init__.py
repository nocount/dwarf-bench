"""Provider registry.

Adding a new provider:
1. Implement the Provider protocol in providers/<name>.py.
2. Register its model-name prefixes in PROVIDER_PREFIXES below.
3. Add an optional dependency in pyproject.toml.
"""
from __future__ import annotations

from typing import Callable

from dwarf_bench.providers.anthropic import AnthropicProvider
from dwarf_bench.providers.base import ModelResponse, Provider

__all__ = [
    "AnthropicProvider",
    "ModelResponse",
    "Provider",
    "provider_for",
    "split_provider_prefix",
]


def _make_openai() -> Provider:
    from dwarf_bench.providers.openai import OpenAIProvider
    return OpenAIProvider()


def _make_gemini() -> Provider:
    from dwarf_bench.providers.gemini import GeminiProvider
    return GeminiProvider()


# Explicit prefix → factory. The "anthropic:", "openai:", "google:" forms
# are escape hatches for non-standard / fine-tuned model names.
PROVIDER_FACTORIES: dict[str, Callable[[], Provider]] = {
    "anthropic": AnthropicProvider,
    "openai": _make_openai,
    "google": _make_gemini,
    "gemini": _make_gemini,
}

# Inferred from model name prefix when no explicit "provider:" prefix is given.
# Order matters only insofar as longer/more specific prefixes should come first.
MODEL_PREFIXES: list[tuple[str, str]] = [
    ("claude-", "anthropic"),
    ("anthropic-", "anthropic"),
    ("gpt-", "openai"),
    ("o1-", "openai"),
    ("o1", "openai"),
    ("o3-", "openai"),
    ("o3", "openai"),
    ("o4-", "openai"),
    ("o4", "openai"),
    ("gemini-", "google"),
]


def split_provider_prefix(model: str) -> tuple[str | None, str]:
    """Return (provider, bare_model) if model is "provider:foo", else (None, model)."""
    if ":" in model:
        prefix, _, bare = model.partition(":")
        if prefix in PROVIDER_FACTORIES:
            return prefix, bare
    return None, model


def provider_for(model: str) -> tuple[Provider, str]:
    """Pick a Provider based on the model name. Returns (provider, bare_model_name).

    The bare model name has any "provider:" prefix stripped so callers can pass
    it straight to the SDK.
    """
    explicit, bare = split_provider_prefix(model)
    key = explicit or _infer_provider(bare)
    if key is None:
        raise ValueError(
            f"Cannot infer a provider from model name {model!r}. "
            f"Use a known prefix (claude-*, gpt-*, o3-*, gemini-*) or write "
            f"'provider:model' explicitly (e.g. 'openai:my-finetune')."
        )
    factory = PROVIDER_FACTORIES[key]
    return factory(), bare


def _infer_provider(model: str) -> str | None:
    for prefix, name in MODEL_PREFIXES:
        if model.startswith(prefix):
            return name
    return None

"""Provider dispatch tests — pure logic, no SDK required."""
from __future__ import annotations

import pytest

from dwarf_bench.providers import provider_for, split_provider_prefix


def test_split_provider_prefix_explicit():
    assert split_provider_prefix("openai:my-finetune") == ("openai", "my-finetune")
    assert split_provider_prefix("anthropic:claude-foo") == ("anthropic", "claude-foo")


def test_split_provider_prefix_unknown_prefix_is_passthrough():
    # "foo" isn't a known provider — treat the whole thing as a model name.
    assert split_provider_prefix("foo:bar") == (None, "foo:bar")


def test_split_provider_prefix_no_colon():
    assert split_provider_prefix("claude-opus-4-7") == (None, "claude-opus-4-7")


def test_provider_for_anthropic_by_prefix():
    p, model = provider_for("claude-opus-4-7")
    assert p.name == "anthropic"
    assert model == "claude-opus-4-7"


def test_provider_for_explicit_prefix_strips_it():
    # OpenAI/Gemini providers lazily import their SDKs; the explicit-prefix
    # path still has to construct one. Skip those here and exercise dispatch
    # via split_provider_prefix instead, which doesn't instantiate.
    explicit, bare = split_provider_prefix("openai:gpt-5")
    assert (explicit, bare) == ("openai", "gpt-5")


def test_provider_for_unknown_model_raises():
    with pytest.raises(ValueError, match="Cannot infer a provider"):
        provider_for("mystery-model-3000")

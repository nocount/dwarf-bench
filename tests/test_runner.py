"""Runner + judge behaviour with a fake provider — no network, no SDKs."""
from __future__ import annotations

from dwarf_bench.dataset import Question
from dwarf_bench.judge import grade_artifact
from dwarf_bench.providers.base import ModelResponse
from dwarf_bench.runner import ANSWER_MAX_TOKENS, run_benchmark


class FakeProvider:
    name = "fake"

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.max_tokens_seen: list[int] = []

    async def generate(self, *, model, system, user, max_tokens=1024):
        self.max_tokens_seen.append(max_tokens)
        text = self.answers.get(user, '{"score": 1.0, "reasoning": "ok"}')
        return ModelResponse(
            text=text, model=model, input_tokens=1, output_tokens=max_tokens, raw={}
        )


def _q(id: str, question: str) -> Question:
    return Question(id=id, setting="s", question=question, answer="a")


async def test_run_passes_answer_token_budget():
    provider = FakeProvider({"q?": "an answer"})
    await run_benchmark(provider=provider, model="m", questions=[_q("a", "q?")])
    assert provider.max_tokens_seen == [ANSWER_MAX_TOKENS]


async def test_empty_response_is_recorded_as_error():
    provider = FakeProvider({"q?": ""})
    artifact = await run_benchmark(
        provider=provider, model="m", questions=[_q("a", "q?")]
    )
    (result,) = artifact.results
    assert result.error is not None and "empty response" in result.error
    assert result.response is not None  # raw kept for debugging


async def test_judge_skips_errored_results():
    provider = FakeProvider({"empty?": "", "full?": "an answer"})
    artifact = await run_benchmark(
        provider=provider,
        model="m",
        questions=[_q("a", "empty?"), _q("b", "full?")],
    )
    await grade_artifact(artifact, judge=provider, judge_model="judge")
    empty, full = artifact.results
    assert empty.grade is None
    assert full.grade is not None and full.grade.score == 1.0
    # The empty answer is excluded from accuracy rather than counted as 0.
    assert artifact.accuracy() == 1.0


class FailingJudge:
    name = "fake"

    async def generate(self, *, model, system, user, max_tokens=1024):
        return ModelResponse(text="", model=model, input_tokens=1, output_tokens=0, raw={})


async def test_judge_failure_leaves_answer_ungraded():
    provider = FakeProvider({"q?": "an answer"})
    artifact = await run_benchmark(
        provider=provider, model="m", questions=[_q("a", "q?")]
    )
    await grade_artifact(artifact, judge=FailingJudge(), judge_model="judge")
    (result,) = artifact.results
    # Not a silent 0.0: stays ungraded so a later `grade` run retries it.
    assert result.grade is None
    assert artifact.accuracy() is None

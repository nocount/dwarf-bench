"""Run a set of questions through a model and collect raw responses."""
from __future__ import annotations

import asyncio

from dwarf_bench.dataset import Question
from dwarf_bench.providers.base import Provider
from dwarf_bench.results import RunArtifact, RunResult

SYSTEM_PROMPT = (
    "You are answering trivia questions about fantasy dwarves drawn from "
    "various books, games, and other media.\n\n"
    "Answer using only your internal knowledge. You have no access to tools, "
    "search, or browsing. Do not attempt to reason about what you would look "
    "up — just answer from memory.\n\n"
    "Give a concise, direct answer. If you genuinely do not know, reply "
    "exactly \"I don't know\" rather than guessing."
)


async def run_benchmark(
    *,
    provider: Provider,
    model: str,
    questions: list[Question],
    concurrency: int = 5,
    system_prompt: str = SYSTEM_PROMPT,
) -> RunArtifact:
    artifact = RunArtifact(
        model=model,
        provider=provider.name,
        system_prompt=system_prompt,
    )
    sem = asyncio.Semaphore(concurrency)

    async def one(q: Question) -> RunResult:
        async with sem:
            try:
                resp = await provider.generate(
                    model=model,
                    system=system_prompt,
                    user=q.question,
                )
                return RunResult(question=q, response=resp)
            except Exception as e:
                return RunResult(question=q, error=f"{type(e).__name__}: {e}")

    artifact.results = await asyncio.gather(*(one(q) for q in questions))
    return artifact

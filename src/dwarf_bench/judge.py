"""Custom LLM-as-judge grader.

Takes a RunArtifact (raw model responses) and produces graded results by
asking a judge model to score each answer against the gold answer.
"""
from __future__ import annotations

import asyncio
import json
import re

from dwarf_bench.providers.base import Provider
from dwarf_bench.results import Grade, RunArtifact, RunResult

DEFAULT_JUDGE_MODEL = "claude-opus-4-7"

JUDGE_SYSTEM = (
    "You are an impartial grader for a trivia benchmark. You compare a model's "
    "answer to a gold-standard answer and assign a score. Respond with a single "
    "JSON object and nothing else."
)

JUDGE_TEMPLATE = """Grade the following trivia answer.

Question: {question}
Gold answer: {gold}
Model's answer: {answer}

Scoring rubric:
- 1.0 — Fully correct. All key facts from the gold answer are present. Minor wording or ordering differences are fine. Additional correct detail is fine.
- 0.5 — Partially correct. Some key facts are right but others are wrong, missing, or the answer is ambiguous.
- 0.0 — Incorrect, contradicts the gold answer, or the model said it does not know.

Respond with a JSON object of exactly this shape:
{{"score": <0.0 | 0.5 | 1.0>, "reasoning": "<one short sentence>"}}
"""


async def grade_artifact(
    artifact: RunArtifact,
    *,
    judge: Provider,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    concurrency: int = 5,
) -> RunArtifact:
    """Grade every (ungraded, non-errored) result in-place and return the artifact."""
    sem = asyncio.Semaphore(concurrency)

    async def grade_one(r: RunResult) -> None:
        if r.grade is not None or r.response is None:
            return
        async with sem:
            try:
                user = JUDGE_TEMPLATE.format(
                    question=r.question.question,
                    gold=r.question.answer,
                    answer=r.response.text,
                )
                resp = await judge.generate(
                    model=judge_model,
                    system=JUDGE_SYSTEM,
                    user=user,
                    max_tokens=300,
                )
                r.grade = _parse_grade(resp.text, judge_model)
            except Exception as e:
                r.grade = Grade(
                    score=0.0,
                    reasoning=f"judge error: {type(e).__name__}: {e}",
                    judge_model=judge_model,
                )

    await asyncio.gather(*(grade_one(r) for r in artifact.results))
    return artifact


def _parse_grade(raw: str, judge_model: str) -> Grade:
    """Extract the JSON object from the judge's reply."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in judge reply: {raw!r}")
    data = json.loads(match.group(0))
    score = float(data["score"])
    if score not in (0.0, 0.5, 1.0):
        raise ValueError(f"invalid score {score!r}; must be 0.0, 0.5, or 1.0")
    return Grade(
        score=score,
        reasoning=str(data.get("reasoning", "")).strip(),
        judge_model=judge_model,
    )

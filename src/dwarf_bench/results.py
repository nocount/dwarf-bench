"""Run artifact schema + JSONL persistence.

A run artifact is a single file under results/ recording one model's answers
(and optional grades) for the full question set. Raw responses are preserved
for reproducibility and re-grading without re-querying the model.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from dwarf_bench.dataset import Question
from dwarf_bench.providers.base import ModelResponse


class Grade(BaseModel):
    score: float
    reasoning: str
    judge_model: str


class RunResult(BaseModel):
    question: Question
    response: ModelResponse | None = None
    error: str | None = None
    grade: Grade | None = None


class RunArtifact(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    model: str
    provider: str
    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    system_prompt: str
    results: list[RunResult] = Field(default_factory=list)

    def save(self, directory: str | Path = "results") -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}__{_slug(self.model)}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            header = self.model_dump(mode="json", exclude={"results"})
            header["_type"] = "header"
            f.write(json.dumps(header) + "\n")
            for r in self.results:
                record = r.model_dump(mode="json")
                record["_type"] = "result"
                f.write(json.dumps(record) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RunArtifact":
        path = Path(path)
        header: dict | None = None
        results: list[RunResult] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                rtype = record.pop("_type", None)
                if rtype == "header":
                    header = record
                elif rtype == "result":
                    results.append(RunResult.model_validate(record))
        if header is None:
            raise ValueError(f"{path}: missing header record")
        return cls(**header, results=results)

    def accuracy(self) -> float | None:
        graded = [r.grade.score for r in self.results if r.grade is not None]
        if not graded:
            return None
        return sum(graded) / len(graded)


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

"""Gold-standard Q&A dataset for dwarf-bench."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class Question(BaseModel):
    id: str
    setting: str
    question: str
    answer: str
    notes: str | None = Field(default=None)


def load_questions(path: str | Path) -> list[Question]:
    """Load questions from a JSONL file. Each line is one Question object."""
    path = Path(path)
    questions: list[Question] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
            q = Question.model_validate(data)
            if q.id in seen_ids:
                raise ValueError(f"{path}:{lineno}: duplicate question id {q.id!r}")
            seen_ids.add(q.id)
            questions.append(q)
    return questions

from pathlib import Path

import pytest

from dwarf_bench.dataset import Question, load_questions


def test_loads_seed_questions():
    path = Path(__file__).parent.parent / "data" / "questions.jsonl"
    questions = load_questions(path)
    assert len(questions) >= 3
    assert all(isinstance(q, Question) for q in questions)
    assert all(q.id and q.question and q.answer for q in questions)


def test_rejects_duplicate_ids(tmp_path):
    f = tmp_path / "dup.jsonl"
    f.write_text(
        '{"id": "a", "setting": "s", "question": "q", "answer": "a"}\n'
        '{"id": "a", "setting": "s", "question": "q2", "answer": "a2"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_questions(f)


def test_rejects_invalid_json(tmp_path):
    f = tmp_path / "bad.jsonl"
    f.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_questions(f)

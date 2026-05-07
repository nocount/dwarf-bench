# dwarf-bench

Benchmark that tests LLM offline knowledge of obscure, specific dwarf facts across fantasy media (Tolkien, D&D, Warhammer, Discworld, etc.).

The pipeline queries models with tools disabled, then grades free-form answers against a gold-standard dataset using a custom LLM-as-judge.

See [`PLAN.md`](PLAN.md) for the full design.

## Install

```bash
python -m venv .venv
.venv/Scripts/activate       # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -e ".[dev]"
cp .env.example .env         # then fill in ANTHROPIC_API_KEY
```

## Usage

Run + grade one or more models in a single shot:

```bash
python -m dwarf_bench bench --models claude-opus-4-7,claude-sonnet-4-6,claude-haiku-4-5-20251001
```

Or run and grade separately:

```bash
python -m dwarf_bench run --model claude-sonnet-4-6
python -m dwarf_bench grade results/<run-file>.jsonl
python -m dwarf_bench report
```

To regenerate the leaderboard JSON consumed by [the website](https://nocount.github.io/dwarf-bench.html):

```bash
python -m dwarf_bench report --json -o leaderboard.json
git add leaderboard.json && git commit -m "Update leaderboard" && git push
```

The site fetches `leaderboard.json` from the `main` branch of this repo at page-load time, so a push is all it takes to publish new numbers.

## Dataset

Questions live in `data/questions.jsonl`, one JSON object per line:

```json
{"id": "q001", "setting": "Tolkien", "question": "...", "answer": "...", "notes": "optional source"}
```

Add your own questions by appending to this file. `id` must be unique.

## Tests

```bash
pytest
```

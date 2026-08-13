# RAG QE Platform

A local, fully-offline **Retrieval-Augmented Generation** application built with a
**Quality Engineering** focus: every answer the pipeline produces can be measured,
gated, and tracked over time using [Ragas](https://docs.ragas.io).

No API keys. Everything runs against local [Ollama](https://ollama.com) models.

## What's in the box

| Layer | What it does |
|---|---|
| **RAG pipeline** | numpy vector store + Ollama embeddings (`nomic-embed-text`) + Ollama generation (`llama3.2:3b`) |
| **REST API** (FastAPI) | `/ingest`, `/query`, `/evaluate`, `/metrics/history`, `/metrics/latest` |
| **Ragas eval harness** | faithfulness, answer relevancy, context precision & recall, scored by a local judge (`gemma4:12b`) against a golden dataset |
| **Quality gates** | a run fails if any mean metric drops below its threshold |
| **Test automation** | offline pytest suite (deterministic `fake` provider) + a gated live eval test, wired into CI |
| **Dashboard** | metrics-over-time chart + latest-run KPI tiles at `http://localhost:8000/` |

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Ollama](https://ollama.com) running locally with these models pulled:
  ```bash
  ollama pull nomic-embed-text   # embeddings
  ollama pull llama3.2:3b        # answer generation
  ollama pull gemma4:12b         # eval judge (must be strong enough for structured output)
  ```

## Quickstart

```bash
make setup     # uv sync — create the environment
make ingest    # build the vector index from data/docs
make serve     # start the API + dashboard on http://localhost:8000
```

Then ask a question:

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"What does faithfulness measure?"}' | jq
```

Run an evaluation and watch the dashboard update:

```bash
make eval                                   # CLI eval over the golden dataset
# or: curl -s -XPOST localhost:8000/evaluate -d '{}' -H 'content-type: application/json'
open http://localhost:8000/
```

## Quality Engineering workflow

The whole point of this repo is that **quality is measured, not assumed.**

- **Golden dataset** — `eval/dataset.py` holds curated question/reference pairs. It's
  the fixed yardstick every pipeline change is scored against.
- **Quality gates** — thresholds in `app/config.py` (overridable via env). A run's
  overall status is `PASS` only if every mean metric clears its gate.
- **Two test layers:**
  ```bash
  make test        # fast, offline, deterministic (fake provider) — runs in CI on every PR
  make test-live   # full suite incl. the real Ragas eval — needs Ollama
  ```
  The offline suite verifies all the plumbing and the gate logic without an LLM, so
  CI stays fast and hermetic. The live suite is the real quality regression check.

## Configuration

Copy `.env.example` to `.env` to override any setting (models, `top_k`, chunk sizes,
gate thresholds). The key one:

- `RAG_LLM_PROVIDER=ollama` — real local models (default)
- `RAG_LLM_PROVIDER=fake` — deterministic, network-free stubs used by the offline tests

## Notes from building this

- **The judge model matters.** `llama3.2:3b` is fine for *generating* answers but too
  weak to be a Ragas *judge* — it returns malformed structured output. `gemma4:12b`
  works reliably. Choosing the judge is itself a quality decision.
- **Ollama speaks OpenAI.** The modern Ragas metrics need an Instructor LLM; we build
  one from an `AsyncOpenAI` client pointed at Ollama's `/v1` endpoint.
- **Pinned langchain.** Ragas 0.4.x still imports paths that langchain 1.x removed, so
  the langchain stack is pinned to the 0.3.x line (see `pyproject.toml`).

## Project layout

```
app/          RAG pipeline + FastAPI service
  config.py     settings & quality-gate thresholds
  models.py     embeddings + generation (ollama | fake)
  vectorstore.py  numpy cosine-similarity store
  ingest.py     load → chunk → embed → persist
  rag.py        retrieve → generate
  api.py        REST endpoints + dashboard
eval/         Ragas evaluation harness
  dataset.py    golden question/reference pairs
  judge.py      builds Ragas metrics on a local judge
  runner.py     runs eval, applies gates, persists results
data/docs/    bundled sample corpus
dashboard/    metrics-over-time UI
tests/        offline unit tests + gated live eval
```

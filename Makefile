.PHONY: setup ingest serve eval test test-live clean

setup:  ## create the environment from pyproject/uv.lock
	uv sync

ingest:  ## build the vector index from data/docs
	uv run python -m app.ingest

serve:  ## run the REST API + dashboard on http://localhost:8000
	uv run uvicorn app.api:app --reload --port 8000

eval:  ## run the Ragas golden-dataset evaluation (needs Ollama)
	uv run python -m eval.runner

test:  ## fast, offline unit tests (fake provider)
	RAG_LLM_PROVIDER=fake uv run pytest

test-live:  ## full suite incl. live Ragas eval (needs Ollama)
	RAG_LLM_PROVIDER=ollama RAGAS_LIVE=1 uv run pytest

clean:  ## remove index, caches and eval results
	rm -rf .data .pytest_cache eval/results/*.json

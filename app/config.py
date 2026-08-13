"""Central configuration.

All values are overridable via environment variables (prefix ``RAG_``) or a
local ``.env`` file. The one setting worth calling out for Quality Engineering
is ``llm_provider``: set it to ``fake`` to run the whole stack deterministically
with no Ollama dependency (used by the offline unit tests / CI).
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # --- model provider -------------------------------------------------
    # "ollama" -> talk to a local Ollama server. "fake" -> deterministic
    # in-process stubs (no network), used by the offline test suite.
    llm_provider: str = "ollama"

    ollama_base_url: str = "http://localhost:11434"
    # Ollama also speaks the OpenAI protocol, which is what Ragas needs.
    openai_base_url: str = "http://localhost:11434/v1"

    embed_model: str = "nomic-embed-text"
    gen_model: str = "llama3.2:3b"
    # The judge must be strong enough for structured output. llama3.2:3b is
    # NOT (it echoes JSON schemas); gemma4:12b works well.
    judge_model: str = "gemma4:12b"

    # --- retrieval ------------------------------------------------------
    top_k: int = 4
    chunk_size: int = 700
    chunk_overlap: int = 120

    # --- paths ----------------------------------------------------------
    data_dir: Path = ROOT / "data" / "docs"
    store_path: Path = ROOT / ".data" / "store.npz"
    results_dir: Path = ROOT / "eval" / "results"

    # --- quality gates --------------------------------------------------
    # A run FAILS if any mean metric drops below its threshold. These are the
    # knobs a QE team tunes as the pipeline matures.
    gate_faithfulness: float = 0.70
    gate_answer_relevancy: float = 0.60
    gate_context_precision: float = 0.60
    gate_context_recall: float = 0.60

    @property
    def gates(self) -> dict[str, float]:
        return {
            "faithfulness": self.gate_faithfulness,
            "answer_relevancy": self.gate_answer_relevancy,
            "context_precision": self.gate_context_precision,
            "context_recall": self.gate_context_recall,
        }


settings = Settings()

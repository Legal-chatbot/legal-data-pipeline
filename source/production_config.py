"""Environment-backed production configuration and validation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ProductionSettings:
    app_env: str = "development"
    log_level: str = "INFO"
    faiss_index_path: Path = Path("processed/legal_chunks.faiss")
    faiss_metadata_path: Path | None = None
    faiss_dimension: int | None = None
    neo4j_uri: str = "neo4j://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    llm_api_key: str | None = None
    llm_api_url: str = "https://api.openai.com/v1/chat/completions"
    llm_model: str = "gpt-4o-mini"
    frontend_origins: str = "http://127.0.0.1:5500,http://localhost:5500"
    context_max_contexts: int = 5
    context_max_characters: int = 12000

    @classmethod
    def from_env(cls) -> "ProductionSettings":
        index = Path(os.getenv("FAISS_INDEX_PATH", "processed/legal_chunks.faiss"))
        metadata = os.getenv("FAISS_METADATA_PATH")
        dimension = os.getenv("FAISS_DIMENSION")
        return cls(
            app_env=os.getenv("APP_ENV", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            faiss_index_path=index,
            faiss_metadata_path=Path(metadata) if metadata else None,
            faiss_dimension=int(dimension) if dimension else None,
            neo4j_uri=os.getenv("NEO4J_URI", cls.neo4j_uri),
            neo4j_user=os.getenv("NEO4J_USER", cls.neo4j_user),
            neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
            neo4j_database=os.getenv("NEO4J_DATABASE", cls.neo4j_database),
            llm_api_key=os.getenv("LLM_API_KEY"),
            llm_api_url=os.getenv("LLM_API_URL", cls.llm_api_url),
            llm_model=os.getenv("LLM_MODEL", cls.llm_model),
            frontend_origins=os.getenv("FRONTEND_ORIGINS", cls.frontend_origins),
            context_max_contexts=int(os.getenv("CONTEXT_MAX_CONTEXTS", "5")),
            context_max_characters=int(os.getenv("CONTEXT_MAX_CHARACTERS", "12000")),
        )

    @property
    def faiss_metadata(self) -> Path:
        return self.faiss_metadata_path or self.faiss_index_path.with_suffix(".metadata.json")

    def validate(self) -> None:
        if self.app_env == "production":
            missing = []
            if not self.neo4j_password:
                missing.append("NEO4J_PASSWORD")
            if not self.llm_api_key:
                missing.append("LLM_API_KEY")
            if missing:
                raise ValueError("Missing production configuration: " + ", ".join(missing))
            if not self.faiss_index_path.exists() or not self.faiss_metadata.exists():
                raise ValueError(f"FAISS index or metadata missing: {self.faiss_index_path}")
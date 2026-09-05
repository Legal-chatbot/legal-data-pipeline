"""Production dependency wiring and lifecycle for the legal RAG system."""

from __future__ import annotations

import logging
from typing import Any

from source.context_builder import ContextBuilder, ContextBuilderConfig
from source.embedding_service import EmbeddingService
from source.faiss_vector_store import FaissVectorStore, VectorStoreConfig
from source.fastapi_rag_api import create_app
from source.hybrid_retrieval import HybridRetrievalEngine, Neo4jGraphRetriever
from source.llm_answer_service import LLMAnswerGenerationService, LLMProviderConfig, OpenAICompatibleProvider
from source.production_config import ProductionSettings
from source.query_understanding import QueryUnderstandingService
from source.rag_orchestrator import LegalRAGService

logger = logging.getLogger(__name__)


class ProductionRuntime:
    def __init__(self, settings: ProductionSettings) -> None:
        self.settings = settings
        self.connection = None
        self.service: LegalRAGService | None = None
        self.vector_store: FaissVectorStore | None = None
        self.started = False

    def start(self) -> None:
        if self.started:
            return
        self.settings.validate()
        logging.basicConfig(
            level=getattr(logging, self.settings.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        from graph_neo4j.knowledge_graph import Neo4jConfig, Neo4jConnection

        self.connection = Neo4jConnection(Neo4jConfig(
            uri=self.settings.neo4j_uri,
            user=self.settings.neo4j_user,
            password=self.settings.neo4j_password,
            database=self.settings.neo4j_database,
        ))
        self.connection.connect()
        if self.settings.faiss_dimension is None:
            raise ValueError("FAISS_DIMENSION is required to load the vector index")
        self.vector_store = FaissVectorStore(VectorStoreConfig(dimension=self.settings.faiss_dimension))
        self.vector_store.load(self.settings.faiss_index_path, self.settings.faiss_metadata)
        embedding = EmbeddingService()
        graph = Neo4jGraphRetriever(self.connection)
        retrieval = HybridRetrievalEngine(
            embedding_service=embedding,
            vector_store=self.vector_store,
            graph_retriever=graph,
            query_understanding=QueryUnderstandingService(),
        )
        context = ContextBuilder(ContextBuilderConfig(
            max_contexts=self.settings.context_max_contexts,
            max_characters=self.settings.context_max_characters,
        ))
        provider = OpenAICompatibleProvider(LLMProviderConfig(
            api_key=self.settings.llm_api_key,
            api_url=self.settings.llm_api_url,
            model=self.settings.llm_model,
        ))
        self.service = LegalRAGService(
            query_understanding=QueryUnderstandingService(),
            retrieval=retrieval,
            context_builder=context,
            answer_generation=LLMAnswerGenerationService(provider),
        )
        self.started = True
        logger.info("Production RAG runtime started")

    def stop(self) -> None:
        if self.connection is not None:
            self.connection.close()
        self.started = False
        self.service = None
        logger.info("Production RAG runtime stopped")

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.started and self.service else "not_ready",
            "runtime_started": self.started,
            "faiss_loaded": bool(self.vector_store and self.vector_store.size >= 0),
            "neo4j_connected": self.connection is not None and getattr(self.connection, "_driver", None) is not None,
        }


def create_production_app(settings: ProductionSettings | None = None):
    from contextlib import asynccontextmanager

    runtime = ProductionRuntime(settings or ProductionSettings.from_env())

    @asynccontextmanager
    async def lifespan(app):
        runtime.start()
        app.state.rag_service = runtime.service
        app.state.health_checker = runtime.health
        try:
            yield
        finally:
            runtime.stop()

    app = create_app(lifespan=lifespan)
    app.state.production_runtime = runtime
    return app


app = create_production_app()
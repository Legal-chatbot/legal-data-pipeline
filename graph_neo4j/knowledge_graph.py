"""Repository and connection layer for the Neo4j legal knowledge graph."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
import os
from typing import Any, Callable, Iterable, Mapping

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError

logger = logging.getLogger(__name__)


class KnowledgeGraphError(RuntimeError):
    """Raised when a knowledge graph operation fails."""


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str = "neo4j://127.0.0.1:7687"
    user: str = "neo4j"
    password: str = ""
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        return cls(
            uri=os.getenv("NEO4J_URI", cls.uri),
            user=os.getenv("NEO4J_USER", cls.user),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE", cls.database),
        )

    def validate(self) -> None:
        if not self.password:
            raise ValueError("NEO4J_PASSWORD must be configured")


class Neo4jConnection:
    """Own one Neo4j driver and expose managed read/write transactions."""

    def __init__(self, config: Neo4jConfig) -> None:
        self.config = config
        self._driver: Driver | None = None

    def connect(self) -> None:
        self.config.validate()
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password),
            )
            try:
                self._driver.verify_connectivity()
            except Exception as exc:
                self.close()
                raise KnowledgeGraphError("could not connect to Neo4j") from exc
            logger.info("Connected to Neo4j at %s", self.config.uri)

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "Neo4jConnection":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    @contextmanager
    def session(self):
        if self._driver is None:
            self.connect()
        assert self._driver is not None
        with self._driver.session(database=self.config.database) as session:
            yield session

    def execute_write(self, work: Callable[..., Any], *args: Any) -> Any:
        try:
            with self.session() as session:
                return session.execute_write(work, *args)
        except Neo4jError as exc:
            raise KnowledgeGraphError("Neo4j write transaction failed") from exc

    def execute_read(self, work: Callable[..., Any], *args: Any) -> Any:
        try:
            with self.session() as session:
                return session.execute_read(work, *args)
        except Neo4jError as exc:
            raise KnowledgeGraphError("Neo4j read transaction failed") from exc


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    title: str | None = None
    so_ky_hieu: str | None = None
    loai_van_ban: str | None = None
    ngay_ban_hanh: str | None = None
    tinh_trang_hieu_luc: str | None = None

    def to_properties(self) -> dict[str, Any]:
        return {
            "id": self.document_id,
            "title": self.title,
            "so_ky_hieu": self.so_ky_hieu,
            "loai_van_ban": self.loai_van_ban,
            "ngay_ban_hanh": self.ngay_ban_hanh,
            "tinh_trang_hieu_luc": self.tinh_trang_hieu_luc,
        }


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    part_index: int = 0
    articles: tuple[str, ...] = field(default_factory=tuple)

    def to_properties(self) -> dict[str, Any]:
        return {
            "id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "part_index": self.part_index,
            "articles": list(self.articles),
        }


DOCUMENT_SCHEMA = (
    "CREATE CONSTRAINT document_id_unique IF NOT EXISTS "
    "FOR (d:Document) REQUIRE d.id IS UNIQUE"
)
CHUNK_SCHEMA = (
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
    "FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
)
DOCUMENT_LOOKUP_INDEX = (
    "CREATE INDEX document_so_ky_hieu IF NOT EXISTS "
    "FOR (d:Document) ON (d.so_ky_hieu)"
)
CHUNK_DOCUMENT_INDEX = (
    "CREATE INDEX chunk_document_id IF NOT EXISTS "
    "FOR (c:Chunk) ON (c.document_id)"
)


class KnowledgeGraphRepository:
    """CRUD and query operations for ``Document`` and ``Chunk`` nodes."""

    ALLOWED_RELATIONSHIPS = frozenset({"AMENDS", "REPLACES", "ABROGATES"})

    def __init__(self, connection: Neo4jConnection) -> None:
        self.connection = connection

    def ensure_schema(self) -> None:
        statements = (
            DOCUMENT_SCHEMA,
            CHUNK_SCHEMA,
            DOCUMENT_LOOKUP_INDEX,
            CHUNK_DOCUMENT_INDEX,
        )
        try:
            with self.connection.session() as session:
                for statement in statements:
                    session.run(statement).consume()
        except Neo4jError as exc:
            raise KnowledgeGraphError("could not initialize Neo4j schema") from exc

    def upsert_document(self, document: DocumentRecord) -> dict[str, Any]:
        query = """
        MERGE (d:Document {id: $document.id})
        SET d.title = $document.title,
            d.so_ky_hieu = $document.so_ky_hieu,
            d.loai_van_ban = $document.loai_van_ban,
            d.ngay_ban_hanh = $document.ngay_ban_hanh,
            d.tinh_trang_hieu_luc = $document.tinh_trang_hieu_luc
        RETURN d { .* } AS document
        """

        def write(tx, value):
            record = tx.run(query, document=value).single()
            return record["document"] if record else value

        return self.connection.execute_write(write, document.to_properties())

    def upsert_chunks(self, chunks: Iterable[ChunkRecord]) -> int:
        rows = [chunk.to_properties() for chunk in chunks]
        if not rows:
            return 0
        query = """
        UNWIND $chunks AS chunk
        MERGE (c:Chunk {id: chunk.id})
        SET c.text = chunk.text,
            c.chunk_index = chunk.chunk_index,
            c.part_index = chunk.part_index,
            c.articles = chunk.articles,
            c.document_id = chunk.document_id
        WITH c, chunk
        MATCH (d:Document {id: chunk.document_id})
        MERGE (c)-[:PART_OF]->(d)
        RETURN count(c) AS count
        """

        def write(tx, values):
            record = tx.run(query, chunks=values).single()
            return int(record["count"]) if record else len(values)

        return self.connection.execute_write(write, rows)

    def create_relationship(
        self,
        source_document_id: str,
        target_document_id: str,
        relationship: str,
    ) -> None:
        relationship = relationship.upper()
        if relationship not in self.ALLOWED_RELATIONSHIPS:
            raise ValueError(f"unsupported legal relationship: {relationship}")
        query = f"""
        MATCH (source:Document {{id: $source_id}})
        MATCH (target:Document {{id: $target_id}})
        MERGE (source)-[:{relationship}]->(target)
        """

        def write(tx, source_id, target_id):
            tx.run(query, source_id=source_id, target_id=target_id).consume()

        self.connection.execute_write(
            write,
            source_document_id,
            target_document_id,
        )

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        query = "MATCH (d:Document {id: $document_id}) RETURN d { .* } AS document"

        def read(tx, value):
            record = tx.run(query, document_id=value).single()
            return record["document"] if record else None

        return self.connection.execute_read(read, document_id)

    def get_chunks(
        self,
        document_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be greater than zero")
        query = """
        MATCH (c:Chunk)-[:PART_OF]->(d:Document {id: $document_id})
        RETURN c { .* } AS chunk
        ORDER BY c.part_index, c.chunk_index
        LIMIT $limit
        """

        def read(tx, value, row_limit):
            return [
                record["chunk"]
                for record in tx.run(query, document_id=value, limit=row_limit)
            ]

        return self.connection.execute_read(read, document_id, limit)
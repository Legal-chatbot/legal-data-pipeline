from contextlib import contextmanager

import pytest

from graph_neo4j.knowledge_graph import (
    ChunkRecord,
    DocumentRecord,
    KnowledgeGraphRepository,
    Neo4jConfig,
    Neo4jConnection,
)


class FakeResult:
    def __init__(self, rows=None):
        self.rows = rows or []

    def single(self):
        return self.rows[0] if self.rows else None

    def __iter__(self):
        return iter(self.rows)

    def consume(self):
        return None


class FakeTransaction:
    def __init__(self):
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))
        if "RETURN d" in query:
            document = params.get("document")
            if document is None:
                document = {"id": params["document_id"]}
            return FakeResult([{"document": document}])
        if "RETURN count(c)" in query:
            return FakeResult([{"count": len(params["chunks"])}])
        if "RETURN c" in query:
            return FakeResult([{"chunk": {"id": "c1"}}])
        return FakeResult()


class FakeSession:
    def __init__(self):
        self.transaction = FakeTransaction()
        self.schema_queries = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def execute_write(self, callback, *args):
        return callback(self.transaction, *args)

    def execute_read(self, callback, *args):
        return callback(self.transaction, *args)

    def run(self, query, **params):
        self.schema_queries.append((query, params))
        return FakeResult()


class FakeDriver:
    def __init__(self):
        self.session_instance = FakeSession()
        self.closed = False

    def verify_connectivity(self):
        return None

    def close(self):
        self.closed = True

    def session(self, database):
        return self.session_instance


class FakeConnection:
    def __init__(self):
        self.session_instance = FakeSession()
        self.writes = []
        self.reads = []

    @contextmanager
    def session(self):
        yield self.session_instance

    def execute_write(self, callback, *args):
        self.writes.append(args)
        return callback(self.session_instance.transaction, *args)

    def execute_read(self, callback, *args):
        self.reads.append(args)
        return callback(self.session_instance.transaction, *args)


def test_connection_requires_password():
    with pytest.raises(ValueError, match="NEO4J_PASSWORD"):
        Neo4jConnection(Neo4jConfig()).connect()


def test_schema_document_chunk_upsert_and_query_use_transactions():
    connection = FakeConnection()
    repository = KnowledgeGraphRepository(connection)
    document = DocumentRecord("doc-1", title="Luật thử nghiệm")
    chunk = ChunkRecord("chunk-1", "doc-1", "Điều 1", 0, articles=("1",))

    repository.ensure_schema()
    result = repository.upsert_document(document)
    assert result["id"] == "doc-1"
    assert repository.upsert_chunks([chunk]) == 1
    assert repository.get_document("doc-1")["id"] == "doc-1"
    assert repository.get_chunks("doc-1") == [{"id": "c1"}]
    assert len(connection.writes) == 2
    assert len(connection.reads) == 2


def test_relationships_are_whitelisted():
    repository = KnowledgeGraphRepository(FakeConnection())

    repository.create_relationship("new", "old", "amends")

    with pytest.raises(ValueError, match="unsupported"):
        repository.create_relationship("new", "old", "RELATED_TO")
from source.context_builder import ContextBuilder, ContextBuilderConfig
from source.embedding_service import LegalChunk
from source.hybrid_retrieval import RetrievalResult


def retrieval(chunks, documents, scores, sources, query=None):
    return RetrievalResult(
        chunks=tuple(chunks),
        documents=tuple(documents),
        scores=tuple(scores),
        retrieval_sources=tuple(sources),
        structured_query=query,
    )


def test_builds_hierarchical_llm_context_with_citations():
    result = retrieval(
        [LegalChunk("Điểm a khoản 2 Điều 5. Nội dung pháp lý.", "c1")],
        [{"id": "d1", "title": "Luật Đất đai", "so_ky_hieu": "01/2024/QH15", "tinh_trang_hieu_luc": "Còn hiệu lực"}],
        [0.91],
        [("graph", "vector")],
    )

    context = ContextBuilder().build(result)

    assert "DOCUMENT: Luật Đất đai" in context.text
    assert "ARTICLE: 5" in context.text
    assert "CLAUSE: 2" in context.text
    assert "POINT: a" in context.text
    assert "TEXT:" in context.text
    assert context.items[0].citation == "Luật Đất đai - Điều 5 khoản 2 điểm a"


def test_deduplicates_and_merges_sources():
    chunk = LegalChunk("Điều 5. Nội dung", "same")
    result = retrieval(
        [chunk, chunk],
        [{"id": "d1"}, {"id": "d1", "title": "Luật thử nghiệm"}],
        [0.4, 0.8],
        [("vector",), ("graph",)],
    )

    context = ContextBuilder().build(result)

    assert len(context.items) == 1
    assert context.items[0].retrieval_source == ("graph", "vector")
    assert context.items[0].document_title == "Luật thử nghiệm"


def test_sorts_by_score_plus_configured_context_priority():
    result = retrieval(
        [LegalChunk("Điều 1 thấp", "low"), LegalChunk("Điều 2 cao", "high")],
        [{"id": "d1"}, {"id": "d2"}],
        [0.8, 0.7],
        [("vector",), ("graph",)],
    )

    context = ContextBuilder(
        ContextBuilderConfig(source_priority={"graph": 0.3})
    ).build(result)

    assert context.items[0].document_id == "d2"


def test_limits_context_count_and_characters_without_splitting_when_possible():
    result = retrieval(
        [
            LegalChunk("Điều 1. Đoạn thứ nhất.\n\nĐoạn thứ hai.", "c1"),
            LegalChunk("Điều 2. Nội dung phụ.", "c2"),
        ],
        [{"id": "d1"}, {"id": "d2"}],
        [1.0, 0.5],
        [("vector",), ("vector",)],
    )
    config = ContextBuilderConfig(max_contexts=1, max_characters=1000)

    context = ContextBuilder(config).build(result)

    assert len(context.items) == 1
    assert context.used_characters <= 1000


def test_token_budget_and_empty_retrieval():
    result = retrieval([], [], [], [])

    context = ContextBuilder(ContextBuilderConfig(max_tokens=10)).build(result)

    assert context.text == ""
    assert context.items == ()
    assert context.estimated_tokens == 0
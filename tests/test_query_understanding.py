import pytest

from source.query_understanding import (
    QueryUnderstandingService,
    RuleBasedQueryUnderstanding,
)


@pytest.fixture
def parser():
    return RuleBasedQueryUnderstanding()


def test_rejects_empty_question(parser):
    with pytest.raises(ValueError):
        parser.understand("  ")


@pytest.mark.parametrize(
    ("question", "article", "clause", "point"),
    [
        ("Điều 5 quy định gì?", "5", None, None),
        ("Điều 5 Luật Doanh nghiệp quy định gì?", "5", None, None),
        ("Khoản 2 Điều 5 quy định gì?", "5", "2", None),
        ("Điểm a khoản 2 Điều 5 quy định gì?", "5", "2", "a"),
    ],
)
def test_extracts_vietnamese_article_references(parser, question, article, clause, point):
    result = parser.understand(question)

    assert result.intent == "article_lookup"
    assert len(result.article_references) == 1
    reference = result.article_references[0]
    assert reference.article == article
    assert reference.clause == clause
    assert reference.point == point


def test_extracts_document_identifier_and_law_name(parser):
    result = parser.understand(
        "Nghị định 01 / 2024 / NĐ-CP quy định về hiệu lực của Luật Đất đai"
    )

    assert result.intent == "validity_lookup"
    assert result.document_identifiers == ("01/2024/NĐ-CP",)
    assert "luật đất đai" in result.normalized_entities["law_name"]
    assert "hiệu lực" in result.legal_terms


def test_detects_intents(parser):
    assert parser.understand("Văn bản nào thay thế văn bản này?").intent == "amendment_lookup"
    assert parser.understand("So sánh Điều 5 và Điều 6").intent == "comparison"
    assert parser.understand("Luật nào áp dụng cho trường hợp này?").intent == "document_lookup"
    assert parser.understand("Thẩm quyền của cơ quan là gì?").intent == "legal_information"
    assert parser.understand("Xin chào").intent == "unknown"


def test_normalization_preserves_vietnamese_meaning(parser):
    result = parser.understand("ĐIỂM A KHOẢN 2 ĐIỀU 5 về ĐỐI TƯỢNG ÁP DỤNG")

    assert result.article_references[0].normalized_value == "điểm a khoản 2 điều 5"
    assert "đối tượng áp dụng" in result.legal_terms
    assert result.to_dict()["original_query"].startswith("ĐIỂM A")


def test_service_can_delegate_to_replacement_backend():
    class Backend:
        def understand(self, question):
            return RuleBasedQueryUnderstanding().understand(question)

    result = QueryUnderstandingService(Backend()).understand("Điều 5")

    assert result.original_query == "Điều 5"
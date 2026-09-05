from source.legal_citation import LegalCitationSystem


class Item:
    chunk_id = "c1"
    text = "Điểm a khoản 2 Điều 5. Nội dung."
    document_id = "d1"
    document_title = "Luật thử nghiệm"
    document_number = "01/2026/QH15"
    validity_status = "Còn hiệu lực"
    article = "5"
    clause = "2"
    point = "a"


class Context:
    items = (Item(),)


def test_citation_is_built_from_retrieved_metadata():
    citations = LegalCitationSystem().from_context(Context())

    citation = citations[0]
    assert citation.citation_id == "C1"
    assert citation.label == "Luật thử nghiệm - Điều 5 khoản 2 điểm a"
    assert citation.source_document.document_number == "01/2026/QH15"
    assert citation.source_chunk.chunk_id == "c1"
    assert citation.source_document.validity_status == "Còn hiệu lực"
    assert citation.is_valid is True
    assert citation.is_trusted is False


def test_validates_only_citations_present_in_context():
    citations = LegalCitationSystem().from_context(Context())

    result = LegalCitationSystem().validate("Kết luận [C1] và [C99].", citations)

    assert result.citations[0].is_trusted is True
    assert result.citations[0].used_in_answer is True
    assert result.invalid_references == ("C99",)
    assert result.citations[1].citation_id == "C99"
    assert result.citations[1].is_valid is False
    assert result.missing_citation is False


def test_missing_llm_citation_is_not_trusted():
    citations = LegalCitationSystem().from_context(Context())

    result = LegalCitationSystem().validate("Kết luận không có nguồn.", citations)

    assert result.missing_citation is True
    assert result.citations[0].is_trusted is False
import json

import pytest

from source.llm_answer_service import (
    LLMAnswerGenerationService,
    LLMProviderConfig,
    LLMProviderError,
    OpenAICompatibleProvider,
    SYSTEM_PROMPT,
)


class FakeProvider:
    def __init__(self, answer="Căn cứ Điều 5 [C1], ..."):
        self.answer = answer
        self.calls = []

    def complete(self, system_prompt, user_prompt, *, timeout_seconds):
        self.calls.append((system_prompt, user_prompt, timeout_seconds))
        return self.answer


class Item:
    citation = "Luật thử nghiệm - Điều 5"
    document_id = "d1"
    document_title = "Luật thử nghiệm"
    document_number = "01/2026/QH15"
    validity_status = "Còn hiệu lực"
    retrieval_score = 0.92
    retrieval_source = ("vector", "graph")


class Context:
    text = "DOCUMENT: Luật thử nghiệm\nARTICLE: 5\nTEXT:\nNội dung pháp lý."
    items = (Item(),)


def test_generates_grounded_answer_with_metadata():
    provider = FakeProvider()
    service = LLMAnswerGenerationService(provider, timeout_seconds=12)

    result = service.generate("Điều 5 quy định gì?", {"intent": "article_lookup"}, Context())

    assert result.answer.startswith("Căn cứ")
    assert result.citations == ("Luật thử nghiệm - Điều 5",)
    assert result.referenced_documents[0]["so_ky_hieu"] == "01/2026/QH15"
    assert result.confidence == 0.92
    assert provider.calls[0][0] == SYSTEM_PROMPT
    assert "Điều 5 quy định gì?" in provider.calls[0][1]
    assert "[C1] Luật thử nghiệm - Điều 5" in provider.calls[0][1]


def test_empty_context_returns_warning_without_calling_provider():
    provider = FakeProvider()
    result = LLMAnswerGenerationService(provider).generate("Câu hỏi", {}, "")

    assert result.confidence == 0.0
    assert result.warnings
    assert provider.calls == []


def test_provider_error_is_wrapped():
    class BrokenProvider:
        def complete(self, *args, **kwargs):
            raise TimeoutError("timed out")

    with pytest.raises(LLMProviderError, match="generation failed"):
        LLMAnswerGenerationService(BrokenProvider()).generate("Câu hỏi", {}, Context())


def test_openai_compatible_provider_uses_api_key_and_retries(monkeypatch):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "Đáp án"}}]}).encode()

    def fake_urlopen(http_request, timeout):
        calls.append((http_request, timeout))
        return Response()

    monkeypatch.setattr("source.llm_answer_service.request.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        LLMProviderConfig(api_key="secret", max_retries=1, retry_backoff_seconds=0)
    )

    assert provider.complete("system", "user", timeout_seconds=3) == "Đáp án"
    assert calls[0][1] == 3
    assert calls[0][0].get_header("Authorization") == "Bearer secret"


def test_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    provider = OpenAICompatibleProvider(LLMProviderConfig(api_key=None))

    with pytest.raises(LLMProviderError, match="missing API key"):
        provider.complete("system", "user", timeout_seconds=1)
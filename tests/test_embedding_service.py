import numpy as np
import pytest

from source.embedding_service import EmbeddingConfig, EmbeddingService, LegalChunk


class FakeModel:
    max_seq_length = 0

    def __init__(self):
        self.calls = []

    def encode(self, sentences, **kwargs):
        self.calls.append((list(sentences), kwargs))
        return np.array(
            [[len(text), text.count("a"), 1.0] for text in sentences],
            dtype=np.float32,
        )


@pytest.fixture(autouse=True)
def clear_embedding_cache():
    EmbeddingService.clear_model_cache()
    yield
    EmbeddingService.clear_model_cache()


def factory_for(model):
    def factory(config, device):
        model.max_seq_length = config.max_seq_length
        model.device = device
        return model

    return factory


def test_empty_input_returns_empty_matrix_without_loading_model():
    calls = []

    def factory(config, device):
        calls.append(True)
        return FakeModel()

    result = EmbeddingService(model_factory=factory).embed_chunks([])

    assert result.shape == (0, 0)
    assert calls == []


def test_single_text_returns_normalized_vector():
    model = FakeModel()
    service = EmbeddingService(
        EmbeddingConfig(batch_size=4, max_seq_length=128),
        model_factory=factory_for(model),
    )

    result = service.embed_text("xin chao")

    assert result.shape == (3,)
    assert np.isclose(np.linalg.norm(result), 1.0)
    assert model.calls[0][1]["batch_size"] == 4
    assert model.max_seq_length == 128


def test_batch_text_accepts_legal_chunks_and_preserves_order():
    model = FakeModel()
    service = EmbeddingService(model_factory=factory_for(model))

    result = service.embed_chunks(
        [LegalChunk("mot"), {"text": "hai"}, LegalChunk("ba")]
    )

    assert result.shape == (3, 3)
    assert model.calls[0][0] == ["mot", "hai", "ba"]


def test_shape_is_deterministic_and_model_is_cached():
    model_creations = []

    def factory(config, device):
        model_creations.append(True)
        return FakeModel()

    config = EmbeddingConfig(model_name="test-model")
    first = EmbeddingService(config, model_factory=factory)
    second = EmbeddingService(config, model_factory=factory)

    assert first.embed_text("a").shape == (3,)
    assert second.embed_text("b").shape == (3,)
    assert len(model_creations) == 1


def test_empty_text_is_rejected():
    service = EmbeddingService(model_factory=factory_for(FakeModel()))

    with pytest.raises(ValueError):
        service.embed_text("  ")

    with pytest.raises(ValueError):
        service.embed_chunks([LegalChunk("")])
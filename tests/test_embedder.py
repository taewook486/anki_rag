"""Tests for embedder.py - BGE-M3 임베딩"""

import pytest
from src.models import Document

try:
    from src.embedder import BGEEmbedder, EmbeddingResult
except ImportError:
    pytest.skip("src.embedder not implemented yet", allow_module_level=True)


class TestBGEEmbedder:
    """BGE-M3 임베딩 테스트"""

    def test_embed_single_document(self):
        """단일 문서 임베딩"""
        embedder = BGEEmbedder()
        doc = Document(word="test", meaning="테스트", source="sentences", deck="테스트")

        result = embedder.embed(doc)

        assert result.dense_vector is not None
        assert len(result.dense_vector) == 1024  # BGE-M3 차원
        assert result.sparse_vector is not None

    def test_embed_batch(self):
        """배치 임베딩"""
        docs = [
            Document(word=f"word{i}", meaning=f"의미{i}", source="test", deck="test")
            for i in range(10)
        ]

        embedder = BGEEmbedder()
        results = embedder.embed_batch(docs)

        assert len(results) == 10
        for result in results:
            assert len(result.dense_vector) == 1024

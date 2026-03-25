"""Tests for indexer.py - Qdrant 인덱싱"""

import pytest
from src.models import Document
from src.embedder import EmbeddingResult

try:
    from src.indexer import QdrantIndexer
except ImportError:
    pytest.skip("src.indexer not implemented yet", allow_module_level=True)


class TestQdrantIndexer:
    """Qdrant 인덱서 테스트"""

    def test_upsert_documents(self):
        """문서 upsert 테스트 (Dense + Sparse)"""
        docs = [
            Document(word="test", meaning="테스트", source="test", deck="test")
        ]
        embeddings = [
            EmbeddingResult(
                dense_vector=[0.1] * 1024,
                sparse_vector={1: 0.5, 2: 0.3},
            )
        ]
        indexer = QdrantIndexer(location=":memory:")
        indexer.create_collection()
        indexer.upsert(docs, embeddings)
        assert indexer.client.count("anki_rag").count == 1

    def test_create_collection(self):
        """컬렉션 생성 테스트 - Dense+Sparse 스키마 확인"""
        indexer = QdrantIndexer(location=":memory:")
        indexer.create_collection(collection_name="test", recreate=True)
        info = indexer.client.get_collection("test")
        assert "dense" in info.config.params.vectors
        assert "sparse" in info.config.params.sparse_vectors

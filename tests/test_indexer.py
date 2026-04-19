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


class TestQdrantIndexerGraphIntegration:
    """QdrantIndexer + WordKnowledgeGraph 통합 테스트 (T4)"""

    def test_indexer_builds_graph_after_indexing(self, tmp_path):
        """Given 3개 문서를 인덱싱할 때,
        When upsert가 성공하면,
        Then indexer.graph에 3개 이상의 노드가 생성된다"""
        docs = [
            Document(word="run", meaning="달리다", source="test", deck="TEST"),
            Document(word="walk", meaning="걷다", source="test", deck="TEST"),
            Document(word="jog", meaning="조깅하다", source="test", deck="TEST"),
        ]
        embeddings = [
            EmbeddingResult(dense_vector=[0.1] * 1024, sparse_vector={1: 0.5})
            for _ in docs
        ]

        graph_path = str(tmp_path / "graph")
        indexer = QdrantIndexer(
            location=":memory:",
            graph_persist_path=graph_path,
        )
        indexer.create_collection()
        indexer.upsert(docs, embeddings)

        assert hasattr(indexer, "graph"), "indexer에 graph 속성이 없습니다"
        assert indexer.graph.node_count() >= 3

    def test_indexer_graph_attribute_exists_without_upsert(self, tmp_path):
        """Given upsert 없이 indexer를 생성할 때,
        When indexer를 초기화하면,
        Then graph 속성이 존재하고 비어 있다"""
        indexer = QdrantIndexer(
            location=":memory:",
            graph_persist_path=str(tmp_path / "empty_graph"),
        )
        assert hasattr(indexer, "graph")
        assert indexer.graph.node_count() == 0

    def test_indexer_graph_save_on_upsert(self, tmp_path):
        """Given graph_persist_path가 지정된 indexer에서 upsert 후,
        When 파일 시스템을 확인하면,
        Then .pkl 파일이 생성된다"""
        import os
        docs = [Document(word="test", meaning="테스트", source="t", deck="T")]
        embeddings = [EmbeddingResult(dense_vector=[0.1] * 1024, sparse_vector={1: 0.5})]

        graph_path = str(tmp_path / "graph")
        indexer = QdrantIndexer(location=":memory:", graph_persist_path=graph_path)
        indexer.create_collection()
        indexer.upsert(docs, embeddings)

        assert os.path.exists(graph_path + ".pkl"), ".pkl 파일이 생성되지 않았습니다"

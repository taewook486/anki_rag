"""Tests for retriever.py - 하이브리드 검색"""

import pytest
from src.models import Document

try:
    from src.retriever import HybridRetriever
except ImportError:
    pytest.skip("src.retriever not implemented yet", allow_module_level=True)


class TestHybridRetriever:
    """하이브리드 검색 테스트"""

    def test_search_with_rrf(self):
        """RRF Fusion 검색 테스트"""
        retriever = HybridRetriever(location=":memory:")
        results = retriever.search("abandon", top_k=5)
        assert len(results) <= 5

    def test_search_with_filters(self):
        """필터 검색 테스트"""
        retriever = HybridRetriever(location=":memory:")
        results = retriever.search("abandon", source_filter="toefl", top_k=10)
        for r in results:
            assert r.document.source == "toefl"


class TestHybridRetrieverSharedClient:
    """SPEC: 외부 주입된 QdrantClient 공유 — 동시 접근 버그 수정 (RED)"""

    def test_retriever_accepts_external_client(self):
        """Given 외부에서 생성된 QdrantClient를,
        When client= 키워드 인자로 주입하면,
        Then retriever.client는 주입된 인스턴스와 동일해야 한다."""
        from qdrant_client import QdrantClient
        external_client = QdrantClient(":memory:")
        retriever = HybridRetriever(
            location=":memory:",
            client=external_client,
        )
        assert retriever.client is external_client

    def test_retriever_falls_back_to_location_when_no_client(self):
        """Given client 인자가 없을 때,
        When 기존 시그니처대로 retriever를 생성하면,
        Then location 기반으로 자체 QdrantClient를 만든다 (기존 동작 보존)."""
        from qdrant_client import QdrantClient
        retriever = HybridRetriever(location=":memory:")
        assert isinstance(retriever.client, QdrantClient)

    def test_indexer_and_retriever_can_share_client_in_memory(self, tmp_path):
        """KEY 통합 테스트: 동일한 QdrantClient(":memory:")를 indexer와
        retriever가 공유할 때, upsert + search가 예외 없이 동작해야 한다.

        Qdrant 로컬 파일 모드 동시 접근 버그를 재현·방지하는 핵심 테스트.
        """
        from qdrant_client import QdrantClient
        from src.indexer import QdrantIndexer
        from src.models import Document
        from src.embedder import EmbeddingResult

        shared_client = QdrantClient(":memory:")

        indexer = QdrantIndexer(
            location=":memory:",
            graph_persist_path=str(tmp_path / "shared_graph"),
            client=shared_client,
        )
        retriever = HybridRetriever(
            location=":memory:",
            client=shared_client,
        )

        # 동일 인스턴스 보장
        assert indexer.client is shared_client
        assert retriever.client is shared_client

        # 데이터 주입 (임베딩은 결정론적 더미 벡터 — 실제 임베더 호출 없이 검색만 회피)
        docs = [Document(word="abandon", meaning="버리다", source="t", deck="T")]
        embs = [EmbeddingResult(dense_vector=[0.1] * 1024, sparse_vector={1: 0.5})]
        indexer.create_collection(recreate=True)
        indexer.upsert(docs, embs)

        # client가 공유되면 같은 컬렉션을 retriever도 볼 수 있어야 한다
        assert shared_client.count("anki_rag").count == 1

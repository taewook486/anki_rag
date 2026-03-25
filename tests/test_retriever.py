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

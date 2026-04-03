"""리트리버 - 하이브리드 검색 (Dense + Sparse + RRF)"""

import logging
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    SparseVector,
)

from src.models import Document, SearchResult
from src.embedder import BGEEmbedder
from src.cache import get_search_cache, make_cache_key

logger = logging.getLogger(__name__)


class HybridRetriever:
    """하이브리드 검색 (Dense + Sparse RRF Fusion)"""

    VECTOR_SIZE = 1024  # BGE-M3 dense 벡터 차원

    def __init__(
        self,
        location: str = "http://localhost:6333",
        collection_name: str = "anki_rag",
        rrf_k: int = 60,
        fetch_multiplier: int = 3,
    ):
        """
        Args:
            location: Qdrant 서버 주소
            collection_name: 컬렉션 이름
            rrf_k: RRF 상수 (기본값 60)
            fetch_multiplier: RRF 후보 풀 확대 배수 — Dense·Sparse 각각
                              top_k * fetch_multiplier 개를 가져온 뒤 RRF 적용
        """
        self.collection_name = collection_name
        self.rrf_k = rrf_k
        self.fetch_multiplier = fetch_multiplier
        if location == ":memory:" or location.startswith("http"):
            self.client = QdrantClient(location=location)
        else:
            self.client = QdrantClient(path=location)
        self._embedder: Optional[BGEEmbedder] = None
        self._ensure_collection()

    @property
    def embedder(self) -> BGEEmbedder:
        """임베더 lazy 로딩"""
        if self._embedder is None:
            self._embedder = BGEEmbedder()
        return self._embedder

    def _ensure_collection(self) -> None:
        """컬렉션이 없으면 Dense+Sparse 스키마로 생성"""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=self.VECTOR_SIZE, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    ),
                },
            )

    # @MX:ANCHOR: 하이브리드 검색 공개 진입점
    # @MX:REASON: [AUTO] RAGPipeline.query(), api/routes/search.py, __main__.py 에서 호출
    def search(
        self,
        query: str,
        top_k: int = 10,
        source_filter: Optional[str] = None,
        deck_filter: Optional[str] = None,
        exclude_sources: Optional[list[str]] = None,
        deduplicate: bool = True,
    ) -> list[SearchResult]:
        """
        하이브리드 검색 (Dense + Sparse RRF Fusion)

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            source_filter: source 필터 (선택)
            deck_filter: deck 필터 (선택)
            exclude_sources: 제외할 source 목록 (예: ["sentences"])
            deduplicate: word 기준 중복 제거 여부

        Returns:
            SearchResult 리스트
        """
        # Level 1 캐시 조회
        cache = get_search_cache()
        cache_key = make_cache_key(
            query=query, top_k=top_k, source_filter=source_filter,
            deck_filter=deck_filter, exclude_sources=exclude_sources,
            deduplicate=deduplicate,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        # 쿼리 임베딩 — query instruction prefix 적용
        query_result = self.embedder.embed_query(query)
        query_dense: list[float] = query_result.dense_vector
        query_sparse = SparseVector(
            indices=list(query_result.sparse_vector.keys()),
            values=list(query_result.sparse_vector.values()),
        )

        # 필터 구성 — source, deck 복합 필터 + exclude 지원
        query_filter = self._build_filter(source_filter, deck_filter, exclude_sources)

        # fetch_multiplier 적용: RRF 후보 풀 확대
        fetch_limit = top_k * self.fetch_multiplier

        # Dense 검색
        dense_response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_dense,
            using="dense",
            limit=fetch_limit,
            query_filter=query_filter,
            with_payload=True,
        )

        # Sparse 검색
        sparse_response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_sparse,
            using="sparse",
            limit=fetch_limit,
            query_filter=query_filter,
            with_payload=True,
        )

        # RRF Fusion으로 병합
        results = self._rrf_fusion(dense_response.points, sparse_response.points, top_k * 3)

        # 정확 매칭 부스팅
        results = self._boost_exact_match(results, query)

        # word 기준 중복 제거
        if deduplicate:
            results = self._deduplicate_by_word(results)

        # 점수 재정렬 후 top_k 반환
        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:top_k]

        # rank 재부여
        for i, r in enumerate(results, 1):
            r.rank = i

        # Level 1 캐시 저장
        cache.set(cache_key, results)

        return results

    def _build_filter(
        self,
        source_filter: Optional[str],
        deck_filter: Optional[str],
        exclude_sources: Optional[list[str]] = None,
    ) -> Optional[Filter]:
        """source·deck 필터를 Qdrant Filter 객체로 변환"""
        conditions = []
        if source_filter:
            conditions.append(
                FieldCondition(key="source", match=MatchValue(value=source_filter))
            )
        if deck_filter:
            conditions.append(
                FieldCondition(key="deck", match=MatchValue(value=deck_filter))
            )

        must_not = []
        for src in (exclude_sources or []):
            must_not.append(
                FieldCondition(key="source", match=MatchValue(value=src))
            )

        if not conditions and not must_not:
            return None
        return Filter(
            must=conditions if conditions else None,
            must_not=must_not if must_not else None,
        )

    def _rrf_fusion(
        self, dense_results: list, sparse_results: list, top_k: int
    ) -> list[SearchResult]:
        """RRF (Reciprocal Rank Fusion) 병합

        score = Σ 1/(rank + k),  k=60
        """
        scores: dict = {}
        payloads: dict = {}

        for rank, hit in enumerate(dense_results, 1):
            doc_id = hit.id
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank + self.rrf_k)
            payloads[doc_id] = hit.payload

        for rank, hit in enumerate(sparse_results, 1):
            doc_id = hit.id
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank + self.rrf_k)
            if doc_id not in payloads:
                payloads[doc_id] = hit.payload

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            SearchResult(
                document=Document(**payloads[doc_id]),
                score=score,
                rank=rank,
            )
            for rank, (doc_id, score) in enumerate(sorted_docs, 1)
        ]

    def _boost_exact_match(
        self, results: list[SearchResult], query: str
    ) -> list[SearchResult]:
        """쿼리와 word 필드가 정확히 일치하면 점수 부스팅"""
        query_lower = query.strip().lower()
        for result in results:
            word_lower = result.document.word.strip().lower()
            if word_lower == query_lower:
                # 정확 일치: 2배 부스팅
                result.score *= 2.0
            elif query_lower in word_lower or word_lower.startswith(query_lower):
                # 부분 일치 (abandon → abandonment): 1.5배 부스팅
                result.score *= 1.5
        return results

    def _deduplicate_by_word(
        self, results: list[SearchResult]
    ) -> list[SearchResult]:
        """동일 word에 대해 최고 점수 결과만 유지"""
        seen: dict[str, SearchResult] = {}
        for result in results:
            word_key = result.document.word.strip().lower()
            if word_key not in seen or result.score > seen[word_key].score:
                seen[word_key] = result
        return list(seen.values())

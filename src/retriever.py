"""리트리버 - 하이브리드 검색 (Dense + Sparse + RRF)"""

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
    ) -> list[SearchResult]:
        """
        하이브리드 검색 (Dense + Sparse RRF Fusion)

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            source_filter: source 필터 (선택)
            deck_filter: deck 필터 (선택)

        Returns:
            SearchResult 리스트
        """
        # 쿼리 임베딩 — query instruction prefix 적용
        query_result = self.embedder.embed_query(query)
        query_dense: list[float] = query_result.dense_vector
        query_sparse = SparseVector(
            indices=list(query_result.sparse_vector.keys()),
            values=list(query_result.sparse_vector.values()),
        )

        # 필터 구성 — source, deck 복합 필터 지원
        query_filter = self._build_filter(source_filter, deck_filter)

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

        # RRF Fusion으로 병합 후 top_k로 자름
        return self._rrf_fusion(dense_response.points, sparse_response.points, top_k)

    def _build_filter(
        self,
        source_filter: Optional[str],
        deck_filter: Optional[str],
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
        if not conditions:
            return None
        return Filter(must=conditions)

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

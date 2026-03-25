"""리트리버 - 하이브리드 검색 (Dense + Sparse + RRF)"""

from typing import Optional
import numpy as np
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
    ):
        """
        Args:
            location: Qdrant 서버 주소
            collection_name: 컬렉션 이름
            rrf_k: RRF 상수 (기본값 60)
        """
        self.collection_name = collection_name
        self.rrf_k = rrf_k
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

    def search(
        self,
        query: str,
        top_k: int = 10,
        source_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        하이브리드 검색 (Dense + Sparse RRF Fusion)

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            source_filter: source 필터 (선택)

        Returns:
            SearchResult 리스트
        """
        # 쿼리 임베딩 (Dense + Sparse 동시)
        query_emb = self.embedder.model.encode(
            [query], return_dense=True, return_sparse=True, return_colbert_vecs=False,
        )

        # Dense 쿼리 벡터: shape (1, 1024) -> list[float]
        query_dense: list[float] = np.array(query_emb["dense_vecs"]).flatten().tolist()

        # Sparse 쿼리 벡터: dict[int, float] -> SparseVector
        sparse_weights = self.embedder._convert_sparse(query_emb["lexical_weights"][0])
        query_sparse = SparseVector(
            indices=list(sparse_weights.keys()),
            values=list(sparse_weights.values()),
        )

        # 필터 구성
        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
            )

        # Dense 검색
        dense_response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_dense,
            using="dense",
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        # Sparse 검색
        sparse_response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_sparse,
            using="sparse",
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        # RRF Fusion으로 병합
        return self._rrf_fusion(dense_response.points, sparse_response.points, top_k)

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

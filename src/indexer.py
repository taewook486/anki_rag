"""인덱서 - Qdrant 벡터 DB 인덱싱"""

import logging
import uuid
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SparseVectorParams,
    SparseIndexParams,
    SparseVector,
)

from src.graph import WordKnowledgeGraph, build_from_documents
from src.models import Document
from src.embedder import EmbeddingResult

logger = logging.getLogger(__name__)

# @MX:NOTE: 대용량 upsert 시 메모리/타임아웃 방지용 배치 크기
_UPSERT_BATCH_SIZE = 500

# 기본 그래프 영속화 경로 (data/graph 기준)
_DEFAULT_GRAPH_PATH = "data/graph"


class QdrantIndexer:
    """Qdrant 인덱서"""

    # @MX:ANCHOR: [AUTO] QdrantIndexer 생성자 — 그래프 통합 진입점
    # @MX:REASON: [AUTO] API, CLI, 테스트 등 복수 호출자에서 사용되는 공개 API
    def __init__(
        self,
        location: str = "http://localhost:6333",
        collection_name: str = "anki_rag",
        vector_size: int = 1024,
        graph_persist_path: Optional[str] = None,
    ):
        """
        Args:
            location: Qdrant 서버 주소 또는 ":memory:" for in-memory
            collection_name: 컬렉션 이름
            vector_size: 벡터 차원 (BGE-M3: 1024)
            graph_persist_path: 그래프 영속화 경로 (확장자 제외). None이면 기본 경로 사용.
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        if location == ":memory:" or location.startswith("http"):
            self.client = QdrantClient(location=location)
        else:
            self.client = QdrantClient(path=location)

        # 그래프 초기화 — 파일이 존재하면 자동 로드
        _graph_path = graph_persist_path or _DEFAULT_GRAPH_PATH
        self._graph_persist_path = _graph_path
        self.graph = WordKnowledgeGraph(persist_path=_graph_path)

    def create_collection(self, collection_name: Optional[str] = None, recreate: bool = False):
        """Dense + Sparse 하이브리드 컬렉션 생성"""
        name = collection_name or self.collection_name

        if recreate and self.client.collection_exists(name):
            self.client.delete_collection(name)

        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config={
                    "dense": VectorParams(size=self.vector_size, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    ),
                },
            )

    def upsert(
        self,
        documents: list[Document],
        embeddings: list[EmbeddingResult],
        batch_size: int = _UPSERT_BATCH_SIZE,
    ):
        """
        문서 upsert (Dense + Sparse 벡터)

        Args:
            documents: Document 리스트
            embeddings: EmbeddingResult 리스트 (dense_vector + sparse_vector)
            batch_size: 배치당 upsert 포인트 수 (기본 500)
        """
        points = []
        for doc, emb in zip(documents, embeddings):
            sparse_indices = list(emb.sparse_vector.keys())
            sparse_values = list(emb.sparse_vector.values())

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": emb.dense_vector,
                        "sparse": SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                    },
                    payload={
                        "word": doc.word,
                        "meaning": doc.meaning,
                        "pronunciation": doc.pronunciation,
                        "example": doc.example,
                        "example_translation": doc.example_translation,
                        "source": doc.source,
                        "deck": doc.deck,
                        "tags": doc.tags,
                        "note_type": doc.note_type,
                        "audio_paths": doc.audio_paths,
                        "difficulty": doc.difficulty,
                        "synonyms": doc.synonyms,
                    },
                )
            )

        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
            )

        # Qdrant 인덱싱 성공 후 지식 그래프 빌드 및 저장
        try:
            build_from_documents(self.graph, documents)
            self.graph.save(self._graph_persist_path)
            logger.info(
                "Indexer: 그래프 빌드 및 저장 완료 (노드: %d, 엣지: %d)",
                self.graph.node_count(),
                self.graph.edge_count(),
            )
        except Exception:
            logger.warning("Indexer: 그래프 빌드/저장 실패 — 계속 진행", exc_info=True)

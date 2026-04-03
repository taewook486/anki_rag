"""인덱서 - Qdrant 벡터 DB 인덱싱"""

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

from src.models import Document
from src.embedder import EmbeddingResult

# @MX:NOTE: 대용량 upsert 시 메모리/타임아웃 방지용 배치 크기
_UPSERT_BATCH_SIZE = 500


class QdrantIndexer:
    """Qdrant 인덱서"""

    def __init__(
        self,
        location: str = "http://localhost:6333",
        collection_name: str = "anki_rag",
        vector_size: int = 1024,
    ):
        """
        Args:
            location: Qdrant 서버 주소 또는 ":memory:" for in-memory
            collection_name: 컬렉션 이름
            vector_size: 벡터 차원 (BGE-M3: 1024)
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        if location == ":memory:" or location.startswith("http"):
            self.client = QdrantClient(location=location)
        else:
            self.client = QdrantClient(path=location)

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

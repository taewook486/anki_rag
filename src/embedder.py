"""임베딩 - BGE-M3 Dense+Sparse 벡터 생성"""

from typing import Optional
from pydantic import BaseModel
import torch

from src.models import Document

# @MX:NOTE: BGE-M3 권장 query instruction — 문서 임베딩에는 적용하지 않음
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingResult(BaseModel):
    """임베딩 결과"""

    dense_vector: list[float]  # 1024차원 Dense 벡터
    sparse_vector: dict[int, float]  # SPLADE Sparse 벡터


class BGEEmbedder:
    """BGE-M3 임베딩 모델 래퍼"""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: Optional[str] = None):
        """
        Args:
            model_name: HuggingFace 모델명
            device: 실행 디바이스 (cuda/mps/cpu, None=자동감지)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """모델 lazy 로딩"""
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            # CPU 환경에서 fp16은 지원되지 않으므로 조건부 비활성화
            use_fp16 = self.device != "cpu"
            self._model = BGEM3FlagModel(
                self.model_name, device=self.device, use_fp16=use_fp16
            )

    @property
    def model(self):
        """모델 로드 프로퍼티"""
        self._load_model()
        return self._model

    def embed(self, doc: Document) -> EmbeddingResult:
        """
        단일 문서 임베딩 (문서 인덱싱용 — instruction 없음)

        Args:
            doc: Document 객체

        Returns:
            EmbeddingResult
        """
        text = self._build_text(doc)

        output = self.model.encode(
            [text],
            batch_size=1,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        return EmbeddingResult(
            dense_vector=output["dense_vecs"][0].tolist(),
            sparse_vector=self._convert_sparse(output["lexical_weights"][0]),
        )

    def embed_batch(self, docs: list[Document]) -> list[EmbeddingResult]:
        """
        배치 임베딩 (문서 인덱싱용 — instruction 없음)

        Args:
            docs: Document 리스트

        Returns:
            EmbeddingResult 리스트
        """
        texts = [self._build_text(doc) for doc in docs]

        outputs = self.model.encode(
            texts,
            batch_size=32,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        results = []
        for i in range(len(texts)):
            results.append(
                EmbeddingResult(
                    dense_vector=outputs["dense_vecs"][i].tolist(),
                    sparse_vector=self._convert_sparse(outputs["lexical_weights"][i]),
                )
            )
        return results

    def embed_query(self, query: str) -> EmbeddingResult:
        """
        검색 쿼리 임베딩 (BGE-M3 query instruction prefix 적용)

        문서 임베딩(embed/embed_batch)과 달리 쿼리에는 instruction을 추가해야
        검색 정확도가 향상된다.

        Args:
            query: 검색 쿼리 문자열

        Returns:
            EmbeddingResult
        """
        text = self._build_query_text(query)

        output = self.model.encode(
            [text],
            batch_size=1,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        return EmbeddingResult(
            dense_vector=output["dense_vecs"][0].tolist(),
            sparse_vector=self._convert_sparse(output["lexical_weights"][0]),
        )

    def _build_text(self, doc: Document) -> str:
        """문서 임베딩 텍스트 구성"""
        parts = [doc.word, doc.meaning]
        if doc.example:
            parts.append(doc.example)
        return " ".join(parts)

    def _build_query_text(self, query: str) -> str:
        """쿼리 임베딩 텍스트 구성 — BGE-M3 권장 instruction prefix 적용"""
        return f"{_QUERY_INSTRUCTION}{query}"

    def _convert_sparse(self, sparse_weights) -> dict[int, float]:
        """Sparse 가중치를 dict로 변환

        BGE-M3의 lexical_weights는 dict[int, float] 형태
        (token_id -> weight 매핑)를 반환함.
        """
        if isinstance(sparse_weights, dict):
            return {int(k): float(v) for k, v in sparse_weights.items() if v > 0}

        # fallback: 리스트/배열 형태
        if hasattr(sparse_weights, "tolist"):
            sparse_list = sparse_weights.tolist()
        else:
            sparse_list = list(sparse_weights)

        return {i: float(w) for i, w in enumerate(sparse_list) if w > 0}

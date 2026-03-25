"""인덱싱 API 라우트 - SPEC-RAG-002 REQ-005 (인덱싱 페이지 백엔드)"""

import os
import threading
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

# 인덱싱 상태 (모듈 수준 공유 상태)
_index_state: dict = {
    "status": "idle",  # idle | running | done | error
    "progress": 0.0,
    "current_file": "",
    "message": "",
    "total": 0,
    "indexed": 0,
    "error": "",
}
_lock = threading.Lock()


class IndexRequest(BaseModel):
    """인덱싱 요청"""
    data_dir: str = Field("./data", description="데이터 디렉토리 경로")
    source: str | None = Field(None, description="특정 source만 인덱싱 (None=전체)")
    recreate: bool = Field(True, description="컬렉션 재생성 여부")


class IndexStatus(BaseModel):
    """인덱싱 상태 응답"""
    status: Literal["idle", "running", "done", "error"]
    progress: float = Field(ge=0.0, le=1.0)
    current_file: str
    message: str
    total: int
    indexed: int
    error: str


def _run_indexing(data_dir: str, source: str | None, recreate: bool) -> None:
    """백그라운드 인덱싱 실행"""
    global _index_state

    def update(status=None, progress=None, current_file=None, message=None,
               total=None, indexed=None, error=None):
        with _lock:
            if status is not None:
                _index_state["status"] = status
            if progress is not None:
                _index_state["progress"] = progress
            if current_file is not None:
                _index_state["current_file"] = current_file
            if message is not None:
                _index_state["message"] = message
            if total is not None:
                _index_state["total"] = total
            if indexed is not None:
                _index_state["indexed"] = indexed
            if error is not None:
                _index_state["error"] = error

    try:
        from src.parser import AnkiParser, TextParser
        from src.embedder import BGEEmbedder
        from src.indexer import QdrantIndexer

        update(status="running", progress=0.0, message="파싱 시작...", error="")

        data_path = Path(data_dir)
        all_documents = []

        # .apkg 파일 수집
        apkg_files = list(data_path.glob("*.apkg"))
        txt_files = list(data_path.glob("*.txt"))
        total_files = len(apkg_files) + len(txt_files)

        if total_files == 0:
            update(status="error", error=f"데이터 파일 없음: {data_dir}")
            return

        for file_idx, apkg_file in enumerate(apkg_files):
            source_name = apkg_file.stem
            if source and source_name != source:
                continue

            update(current_file=apkg_file.name,
                   message=f"파싱 중: {apkg_file.name}",
                   progress=file_idx / total_files * 0.4)

            parser = AnkiParser()
            try:
                docs = parser.parse_file(str(apkg_file), source=source_name)
                all_documents.extend(docs)
                update(message=f"{apkg_file.name}: {len(docs)}개 파싱 완료")
            except Exception as e:
                update(message=f"{apkg_file.name}: 파싱 실패 ({e})")

        for file_idx, txt_file in enumerate(txt_files):
            update(current_file=txt_file.name,
                   message=f"파싱 중: {txt_file.name}",
                   progress=(len(apkg_files) + file_idx) / total_files * 0.4)
            parser = TextParser()
            docs = parser.parse_file(str(txt_file), source="sentences", deck="원서 1만 문장")
            all_documents.extend(docs)
            update(message=f"{txt_file.name}: {len(docs)}개 파싱 완료")

        update(total=len(all_documents), indexed=0,
               message=f"총 {len(all_documents)}개 문서 임베딩 시작...",
               progress=0.4)

        if not all_documents:
            update(status="error", error="파싱된 문서 없음")
            return

        # 임베딩 (배치 단위 진행률 업데이트)
        embedder = BGEEmbedder()
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(all_documents), batch_size):
            batch = all_documents[i:i + batch_size]
            from src.embedder import EmbeddingResult
            import numpy as np

            texts = [embedder._build_text(d) for d in batch]
            output = embedder.model.encode(
                texts, batch_size=batch_size,
                return_dense=True, return_sparse=True, return_colbert_vecs=False
            )
            for j in range(len(batch)):
                all_embeddings.append(EmbeddingResult(
                    dense_vector=output["dense_vecs"][j].tolist(),
                    sparse_vector=embedder._convert_sparse(output["lexical_weights"][j]),
                ))

            indexed_so_far = min(i + batch_size, len(all_documents))
            embed_progress = 0.4 + (indexed_so_far / len(all_documents)) * 0.4
            update(indexed=indexed_so_far, progress=embed_progress,
                   message=f"임베딩 중: {indexed_so_far}/{len(all_documents)}")

        update(message="Qdrant 인덱싱 중...", progress=0.85)

        qdrant_location = os.getenv("QDRANT_LOCATION", "./qdrant_data")
        indexer = QdrantIndexer(location=qdrant_location)
        indexer.create_collection(recreate=recreate)
        indexer.upsert(all_documents, all_embeddings)

        update(status="done", progress=1.0, indexed=len(all_documents),
               message=f"인덱싱 완료: {len(all_documents)}개 문서")

    except Exception as e:
        update(status="error", error=str(e), message="인덱싱 실패")


@router.post("/index")
async def start_indexing(request: IndexRequest) -> dict:
    """
    인덱싱 시작 (백그라운드 실행)

    - 이미 실행 중이면 409 반환
    - 상태는 GET /api/index/status로 조회
    """
    with _lock:
        if _index_state["status"] == "running":
            raise HTTPException(status_code=409, detail="이미 인덱싱 중입니다")
        _index_state.update({
            "status": "running", "progress": 0.0,
            "current_file": "", "message": "시작 중...",
            "total": 0, "indexed": 0, "error": "",
        })

    thread = threading.Thread(
        target=_run_indexing,
        args=(request.data_dir, request.source, request.recreate),
        daemon=True,
    )
    thread.start()
    return {"message": "인덱싱 시작됨"}


@router.get("/index/status", response_model=IndexStatus)
async def get_index_status() -> IndexStatus:
    """인덱싱 진행 상태 조회"""
    with _lock:
        state = _index_state.copy()
    return IndexStatus(**state)

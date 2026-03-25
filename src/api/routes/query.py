"""RAG 질의 API 라우트 - RAGPipeline 연동"""

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

from src.retriever import HybridRetriever
from src.rag import RAGPipeline

router = APIRouter()

# 전역 인스턴스 (싱글톤)
_retriever: Optional[HybridRetriever] = None
_rag: Optional[RAGPipeline] = None


def get_rag() -> RAGPipeline:
    """RAGPipeline 인스턴스 반환 (lazy initialization)"""
    global _retriever, _rag
    if _rag is None:
        location = os.getenv("QDRANT_LOCATION", "./qdrant_data")
        _retriever = HybridRetriever(location=location)
        _rag = RAGPipeline(retriever=_retriever)
    return _rag


class QueryRequest(BaseModel):
    """RAG 질의 요청 모델"""
    question: str = Field(..., description="질문", min_length=1)
    top_k: int = Field(5, description="검색할 문서 수", ge=1, le=20)
    source_filter: Optional[str] = Field(None, description="source 필터")


class SourceInfo(BaseModel):
    """출처 정보"""
    word: str
    source: str
    deck: str


class QueryResponse(BaseModel):
    """RAG 질의 응답 모델"""
    answer: str
    sources: List[SourceInfo]


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    RAG 질의 수행

    - **question**: 질문
    - **top_k**: 검색할 문서 수 (1-20)
    - **source_filter**: source 필터 (선택)
    """
    try:
        rag = get_rag()
        
        # RAG 질의 수행
        answer = rag.query(
            question=request.question,
            top_k=request.top_k,
            source_filter=request.source_filter
        )

        # 검색 결과로부터 출처 추출
        search_results = rag.retriever.search(
            query=request.question,
            top_k=request.top_k,
            source_filter=request.source_filter
        )
        
        sources = [
            SourceInfo(
                word=r.document.word,
                source=r.document.source,
                deck=r.document.deck or ""
            )
            for r in search_results
        ]

        return QueryResponse(answer=answer, sources=sources)

    except ValueError as e:
        # ANTHROPIC_API_KEY 누락 등
        raise HTTPException(status_code=500, detail=f"설정 오류: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질의 실패: {str(e)}")

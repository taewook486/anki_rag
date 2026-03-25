"""검색 API 라우트 - HybridRetriever 연동"""

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

from src.retriever import HybridRetriever

router = APIRouter()

# 전역 Retriever 인스턴스 (싱글톤)
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """Retriever 인스턴스 반환 (lazy initialization)"""
    global _retriever
    if _retriever is None:
        location = os.getenv("QDRANT_LOCATION", "./qdrant_data")
        _retriever = HybridRetriever(location=location)
    return _retriever


class SearchRequest(BaseModel):
    """검색 요청 모델"""
    query: str = Field(..., description="검색어", min_length=1)
    top_k: int = Field(10, description="반환할 결과 수", ge=1, le=50)
    source_filter: Optional[str] = Field(None, description="source 필터")


class SearchResultItem(BaseModel):
    """검색 결과 아이템"""
    word: str
    meaning: str
    pronunciation: Optional[str] = None
    example: Optional[str] = None
    example_translation: Optional[str] = None
    source: str
    deck: str
    score: float = Field(..., ge=0.0)
    rank: int
    audio_available: bool


class SearchResponse(BaseModel):
    """검색 응답 모델"""
    results: List[SearchResultItem]


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """
    하이브리드 검색 수행

    - **query**: 검색어
    - **top_k**: 반환할 결과 수 (1-50)
    - **source_filter**: source 필터 (선택)
    """
    try:
        retriever = get_retriever()
        results = retriever.search(
            query=request.query,
            top_k=request.top_k,
            source_filter=request.source_filter
        )

        # SearchResult → SearchResultItem 변환
        search_results = []
        for result in results:
            doc = result.document
            search_results.append(SearchResultItem(
                word=doc.word,
                meaning=doc.meaning,
                pronunciation=doc.pronunciation,
                example=doc.example,
                example_translation=doc.example_translation,
                source=doc.source,
                deck=doc.deck,
                score=result.score,
                rank=result.rank,
                audio_available=doc.audio_path is not None,
            ))

        return SearchResponse(results=search_results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")

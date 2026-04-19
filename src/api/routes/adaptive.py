"""Adaptive RAG API 라우트

설계서 13.4 — POST /api/adaptive 엔드포인트:
    쿼리 복잡도를 자동 분류하여 최적 검색 전략으로 응답
"""

import os
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.adaptive import AdaptiveRAG, AdaptiveResult, QueryComplexity
from src.agent import LearningAgent
from src.rag import RAGPipeline
from src.retriever import HybridRetriever

router = APIRouter()

# 전역 싱글톤
_adaptive: Optional[AdaptiveRAG] = None


def get_adaptive() -> AdaptiveRAG:
    """AdaptiveRAG 인스턴스 반환 (lazy initialization)"""
    global _adaptive
    if _adaptive is None:
        location = os.getenv("QDRANT_LOCATION", "./qdrant_data")
        retriever = HybridRetriever(location=location)
        rag = RAGPipeline(retriever=retriever)
        agent = LearningAgent(retriever=retriever, rag=rag)
        _adaptive = AdaptiveRAG(retriever=retriever, rag=rag, agent=agent)
    return _adaptive


class AdaptiveRequest(BaseModel):
    """Adaptive RAG 요청"""

    question: str = Field(..., description="질문", min_length=1)
    source_filter: Optional[str] = Field(None, description="source 필터")
    deck_filter: Optional[str] = Field(None, description="deck 필터")
    use_graph: bool = Field(True, description="Complex 전략에서 GraphRAG Fusion 사용 여부 (S2)")


class AgentStepInfo(BaseModel):
    """Agent 스텝 요약 (Complex 전략 시)"""

    thought: str
    tool: str
    args: dict[str, Any]
    observation: str
    retry_count: int


class SourceInfo(BaseModel):
    """검색 출처 정보"""

    word: str
    source: str
    deck: str


class AdaptiveResponse(BaseModel):
    """Adaptive RAG 응답"""

    answer: str
    complexity: str = Field(..., description="쿼리 복잡도 (simple/moderate/complex)")
    strategy_used: str = Field(..., description="사용된 전략 (dense_only/hybrid_rrf/agent_react)")
    sources: list[SourceInfo] = Field(default_factory=list)
    agent_steps: Optional[list[AgentStepInfo]] = None
    total_agent_steps: Optional[int] = None


@router.post("/adaptive", response_model=AdaptiveResponse)
async def adaptive_query(request: AdaptiveRequest) -> AdaptiveResponse:
    """
    Adaptive RAG — 쿼리 복잡도 기반 자동 전략 선택

    - **Simple**: 단일 단어 조회 → Dense 검색
    - **Moderate**: 의미 관계 질의 → Hybrid RRF 검색
    - **Complex**: 다단계 추론 → Agent ReAct 루프

    기존 `/api/query`, `/api/agent`를 통합하는 지능형 진입점입니다.
    """
    try:
        adaptive = get_adaptive()
        result: AdaptiveResult = adaptive.query(
            question=request.question,
            source_filter=request.source_filter,
            deck_filter=request.deck_filter,
            use_graph=request.use_graph,
        )

        # 검색 결과에서 출처 정보 추출
        sources = []
        for sr in result.search_results:
            sources.append(SourceInfo(
                word=sr.document.word,
                source=sr.document.source,
                deck=sr.document.deck,
            ))

        # Agent 스텝 정보 (Complex 전략 시)
        agent_steps = None
        total_agent_steps = None
        if result.agent_result is not None:
            total_agent_steps = result.agent_result.total_steps
            agent_steps = [
                AgentStepInfo(
                    thought=s.thought,
                    tool=s.tool,
                    args=s.args,
                    observation=s.observation,
                    retry_count=s.retry_count,
                )
                for s in result.agent_result.steps
            ]

        return AdaptiveResponse(
            answer=result.answer,
            complexity=result.complexity.value,
            strategy_used=result.strategy_used,
            sources=sources,
            agent_steps=agent_steps,
            total_agent_steps=total_agent_steps,
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"설정 오류: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Adaptive RAG 실행 실패: {e}")

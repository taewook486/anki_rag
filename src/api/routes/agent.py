"""Agentic AI Agent API 라우트

설계서 12.5 — POST /api/agent 엔드포인트:
    LearningAgent ReAct 루프를 통해 멀티스텝 질의응답 수행
"""

import os
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agent import AgentResult, LearningAgent
from src.rag import RAGPipeline
from src.retriever import HybridRetriever

router = APIRouter()

# 전역 싱글톤
_agent: Optional[LearningAgent] = None


def get_agent() -> LearningAgent:
    """LearningAgent 인스턴스 반환 (lazy initialization)"""
    global _agent
    if _agent is None:
        location = os.getenv("QDRANT_LOCATION", "./qdrant_data")
        retriever = HybridRetriever(location=location)
        rag = RAGPipeline(retriever=retriever)
        _agent = LearningAgent(retriever=retriever, rag=rag)
    return _agent


class AgentRequest(BaseModel):
    """Agent 질의 요청"""

    question: str = Field(..., description="질문", min_length=1)
    max_steps: int = Field(5, description="최대 ReAct 스텝 수", ge=1, le=10)


class AgentStepResponse(BaseModel):
    """ReAct 스텝 응답"""

    thought: str
    tool: str
    args: dict[str, Any]
    observation: str
    retry_count: int


class AgentResponse(BaseModel):
    """Agent 최종 응답"""

    answer: str
    total_steps: int
    steps: list[AgentStepResponse]


@router.post("/agent", response_model=AgentResponse)
async def run_agent(request: AgentRequest) -> AgentResponse:
    """
    Agentic AI Agent 실행 (ReAct 패턴)

    - **question**: 사용자 질문 (다단계 추론이 필요한 복잡한 질문에 적합)
    - **max_steps**: 최대 ReAct 스텝 수 (기본 5)

    일반 단순 질의는 `/api/query`를 사용하세요.
    """
    try:
        agent = get_agent()
        # max_steps 동적 설정
        agent.max_steps = request.max_steps

        result: AgentResult = agent.run(question=request.question)

        return AgentResponse(
            answer=result.answer,
            total_steps=result.total_steps,
            steps=[
                AgentStepResponse(
                    thought=s.thought,
                    tool=s.tool,
                    args=s.args,
                    observation=s.observation,
                    retry_count=s.retry_count,
                )
                for s in result.steps
            ],
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"설정 오류: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 실행 실패: {e}")

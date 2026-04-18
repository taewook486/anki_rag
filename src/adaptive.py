"""Adaptive RAG — 쿼리 복잡도 기반 검색 전략 동적 선택 (v1.3)

설계서 섹션 13.4 구현:
- 쿼리 복잡도 분류: Simple / Moderate / Complex
- 전략 자동 분기:
    Simple   → Dense-only 검색, top_k=5
    Moderate → Hybrid RRF 검색, top_k=10
    Complex  → LearningAgent (ReAct 루프) 위임
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Optional

from src.agent import AgentResult, LearningAgent
from src.rag import RAGPipeline
from src.retriever import HybridRetriever

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────
# 쿼리 복잡도 분류
# ───────────────────────────────────────────
class QueryComplexity(str, Enum):
    """쿼리 복잡도 등급"""

    SIMPLE = "simple"       # 단일 단어 검색
    MODERATE = "moderate"   # 의미 관계 질의
    COMPLEX = "complex"     # 다단계 추론


# LLM 기반 복잡도 분류 프롬프트
_CLASSIFY_PROMPT = (
    "다음 질문의 복잡도를 분류하세요.\n\n"
    "SIMPLE: 단일 단어 뜻 조회, 발음, 간단한 검색 (예: 'abandon의 뜻은?', 'run')\n"
    "MODERATE: 유의어/반의어 비교, 특정 주제 필터 검색, 예문 요청 "
    "(예: 'give up 유의어', 'TOEFL 빈출 단어')\n"
    "COMPLEX: 여러 단계 추론, 비교 분석, 학습 계획, 난이도별 정리 "
    "(예: 'TOEFL 경제 단어를 난이도별로 정리해줘', "
    "'abandon과 forsake의 뉘앙스 차이를 예문과 함께 비교해줘')\n\n"
    "질문: {question}\n\n"
    "SIMPLE, MODERATE, COMPLEX 중 하나만 답하세요."
)

# 휴리스틱 기반 패턴 (LLM 호출 전 빠른 분류)
_SIMPLE_PATTERNS = [
    # 단일 영어 단어만 (1~3단어, 한글 없음)
    r"^[a-zA-Z][a-zA-Z\s'-]{0,30}$",
    # "X의 뜻은?", "X 뜻" 패턴
    r"^[a-zA-Z\s'-]+[의의]?\s*뜻",
    # "X 발음" 패턴
    r"^[a-zA-Z\s'-]+\s*발음",
]

_COMPLEX_PATTERNS = [
    # 난이도별, 단계별, 분류 요청
    r"난이도[별순]",
    r"단계[별순]",
    r"정리해\s*줘",
    r"비교.*예문",
    r"학습\s*계획",
    r"차이.*비교",
]


def classify_query_heuristic(question: str) -> Optional[QueryComplexity]:
    """휴리스틱 기반 빠른 분류 (LLM 호출 없이)

    명확한 패턴이면 즉시 분류, 불확실하면 None 반환.
    """
    stripped = question.strip()

    # Simple 패턴 매칭
    for pattern in _SIMPLE_PATTERNS:
        if re.match(pattern, stripped, re.IGNORECASE):
            return QueryComplexity.SIMPLE

    # Complex 패턴 매칭
    for pattern in _COMPLEX_PATTERNS:
        if re.search(pattern, stripped):
            return QueryComplexity.COMPLEX

    return None


def classify_query_llm(
    question: str,
    provider,
    model: str,
) -> QueryComplexity:
    """LLM 기반 복잡도 분류

    Args:
        question: 사용자 질문
        provider: LLMProvider 인스턴스
        model: LLM 모델명

    Returns:
        QueryComplexity 등급
    """
    try:
        resp = provider.generate(
            messages=[
                {"role": "system", "content": "쿼리 복잡도 분류 전문가입니다. SIMPLE, MODERATE, COMPLEX 중 하나만 답하세요."},
                {"role": "user", "content": _CLASSIFY_PROMPT.format(question=question)},
            ],
            model=model,
            max_tokens=10,
        )
        upper = resp.strip().upper()
        if "SIMPLE" in upper:
            return QueryComplexity.SIMPLE
        if "COMPLEX" in upper:
            return QueryComplexity.COMPLEX
        return QueryComplexity.MODERATE
    except Exception:
        logger.warning("LLM 복잡도 분류 실패 — 기본값 MODERATE")
        return QueryComplexity.MODERATE


def classify_query(
    question: str,
    provider=None,
    model: str = "",
) -> QueryComplexity:
    """쿼리 복잡도 분류 (휴리스틱 우선, 불확실 시 LLM)

    Args:
        question: 사용자 질문
        provider: LLMProvider (None이면 휴리스틱만 사용)
        model: LLM 모델명

    Returns:
        QueryComplexity 등급
    """
    # 1단계: 휴리스틱
    heuristic = classify_query_heuristic(question)
    if heuristic is not None:
        logger.info("Adaptive RAG: 휴리스틱 분류 → %s", heuristic.value)
        return heuristic

    # 2단계: LLM 분류
    if provider is not None:
        result = classify_query_llm(question, provider, model)
        logger.info("Adaptive RAG: LLM 분류 → %s", result.value)
        return result

    # 폴백: provider 없으면 MODERATE
    return QueryComplexity.MODERATE


# ───────────────────────────────────────────
# Adaptive RAG 파이프라인
# ───────────────────────────────────────────
class AdaptiveRAG:
    """쿼리 복잡도에 따라 검색 전략을 동적으로 선택하는 RAG 파이프라인

    설계서 13.4:
        Simple   → Dense 검색만, top_k=5
        Moderate → Hybrid 검색, top_k=10, RRF
        Complex  → Agent Loop (ReAct)
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        rag: RAGPipeline,
        agent: LearningAgent,
    ) -> None:
        self.retriever = retriever
        self.rag = rag
        self.agent = agent

    def query(
        self,
        question: str,
        source_filter: Optional[str] = None,
        deck_filter: Optional[str] = None,
        stream: bool = False,
        on_chunk=None,
    ) -> AdaptiveResult:
        """Adaptive RAG 질의

        Args:
            question: 사용자 질문
            source_filter: source 필터
            deck_filter: deck 필터
            stream: 스트리밍 여부 (Simple/Moderate만 지원)
            on_chunk: 스트리밍 콜백

        Returns:
            AdaptiveResult (answer + complexity + strategy_used)
        """
        # 복잡도 분류
        complexity = classify_query(
            question,
            provider=self.rag.provider,
            model=self.rag.model,
        )

        if complexity == QueryComplexity.SIMPLE:
            return self._execute_simple(question, source_filter, deck_filter)

        if complexity == QueryComplexity.COMPLEX:
            return self._execute_complex(question)

        # MODERATE (기본)
        return self._execute_moderate(
            question, source_filter, deck_filter, stream, on_chunk,
        )

    def _execute_simple(
        self,
        question: str,
        source_filter: Optional[str],
        deck_filter: Optional[str],
    ) -> AdaptiveResult:
        """Simple 전략: Dense-only 검색 → RAG 응답"""
        logger.info("Adaptive RAG: Simple 전략 실행")

        search_query = self.rag._extract_search_query(question)

        results = self.retriever.search_dense_only(
            search_query,
            top_k=5,
            source_filter=source_filter,
            deck_filter=deck_filter,
            exclude_sources=["sentences"],
        )
        self.rag.last_results = results

        context = self.rag._build_context(results)
        messages: list[dict] = [
            {"role": "system", "content": self.rag.SYSTEM_PROMPT if hasattr(self.rag, 'SYSTEM_PROMPT') else "당신은 영어 학습 전문가입니다."},
            {"role": "user", "content": self.rag._build_user_content(question, context)},
        ]

        # SYSTEM_PROMPT는 모듈 레벨 상수
        from src.rag import SYSTEM_PROMPT
        messages[0]["content"] = SYSTEM_PROMPT

        answer = self.rag.provider.generate(
            messages=messages,
            model=self.rag.model,
            max_tokens=self.rag.max_tokens,
        )

        return AdaptiveResult(
            answer=answer,
            complexity=QueryComplexity.SIMPLE,
            strategy_used="dense_only",
            search_results=results,
        )

    def _execute_moderate(
        self,
        question: str,
        source_filter: Optional[str],
        deck_filter: Optional[str],
        stream: bool,
        on_chunk,
    ) -> AdaptiveResult:
        """Moderate 전략: Hybrid RRF 검색 → RAG 응답 (기존 파이프라인)"""
        logger.info("Adaptive RAG: Moderate 전략 실행")

        answer = self.rag.query(
            question=question,
            top_k=10,
            source_filter=source_filter,
            deck_filter=deck_filter,
            stream=stream,
            on_chunk=on_chunk,
        )

        return AdaptiveResult(
            answer=answer,
            complexity=QueryComplexity.MODERATE,
            strategy_used="hybrid_rrf",
            search_results=self.rag.last_results,
        )

    def _execute_complex(self, question: str) -> AdaptiveResult:
        """Complex 전략: LearningAgent (ReAct 루프) 위임"""
        logger.info("Adaptive RAG: Complex 전략 실행 (Agent)")

        agent_result: AgentResult = self.agent.run(question)

        return AdaptiveResult(
            answer=agent_result.answer,
            complexity=QueryComplexity.COMPLEX,
            strategy_used="agent_react",
            agent_result=agent_result,
        )


# ───────────────────────────────────────────
# 결과 데이터 클래스
# ───────────────────────────────────────────
class AdaptiveResult:
    """Adaptive RAG 실행 결과"""

    def __init__(
        self,
        answer: str,
        complexity: QueryComplexity,
        strategy_used: str,
        search_results: list | None = None,
        agent_result: AgentResult | None = None,
    ) -> None:
        self.answer = answer
        self.complexity = complexity
        self.strategy_used = strategy_used
        self.search_results = search_results or []
        self.agent_result = agent_result

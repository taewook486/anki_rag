"""Agentic AI Agent — ReAct 패턴 기반 영어 학습 에이전트 (v1.2)

설계서 섹션 12 구현:
- ReAct(Reasoning + Acting) 루프
- 6종 Tool: search_word, rag_query, get_related_words,
             filter_by_source, play_audio, create_study_plan
- Self-Correction: RRF 점수 임계값 미만 시 쿼리 재작성 재시도
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from src.models import Document, SearchResult
from src.rag import RAGPipeline, _MIN_RRF_SCORE
from src.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────
# ReAct 시스템 프롬프트
# ───────────────────────────────────────────
_REACT_SYSTEM_PROMPT = """당신은 영어 학습 AI 에이전트입니다. 도구를 사용해 사용자의 질문에 단계적으로 답변합니다.

[사용 가능한 도구]
- search_word(query, top_k=5, source=null): 하이브리드 벡터 검색 (모든 덱)
- rag_query(question, source=null): RAG 기반 자연어 응답 생성
- get_related_words(word, top_k=5): 유의어·파생어 검색
- filter_by_source(source, query, top_k=5): 특정 소스 필터 검색 (toefl/xfer/phrasal/hacker_toeic/hacker_green/sentences)

[응답 형식 — 반드시 준수]
도구 사용 시:
Thought: <추론>
Action: {"tool": "<tool_name>", "args": {<key>: <value>}}

최종 답변 시:
Final Answer: <답변>

[규칙]
- Action은 반드시 유효한 JSON이어야 합니다.
- 이전 Observation으로 충분한 정보가 수집되면 Final Answer를 출력합니다.
- 최대 {max_steps}번 이내에 Final Answer를 출력해야 합니다."""

_MAX_QUERY_RETRY = 2  # Self-Correction 최대 재시도 횟수


# ───────────────────────────────────────────
# 데이터 클래스
# ───────────────────────────────────────────
@dataclass
class AgentStep:
    """ReAct 루프 1회 스텝 기록"""

    thought: str
    tool: str
    args: dict[str, Any]
    observation: str
    retry_count: int = 0


@dataclass
class AgentResult:
    """에이전트 최종 결과"""

    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_steps: int = 0


# ───────────────────────────────────────────
# LearningAgent
# ───────────────────────────────────────────
class LearningAgent:
    """ReAct 패턴 기반 영어 학습 에이전트

    설계서 12.2 — Agent 구조:
        Thought → Action → Observation 루프
        충분한 정보 수집 시 Final Answer 반환
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        rag: RAGPipeline,
        max_steps: int = 5,
    ) -> None:
        self.retriever = retriever
        self.rag = rag
        self.max_steps = max_steps

        # 설계서 12.3 — Tool 정의
        self._tools: dict[str, Any] = {
            "search_word": self._tool_search_word,
            "rag_query": self._tool_rag_query,
            "get_related_words": self._tool_get_related_words,
            "filter_by_source": self._tool_filter_by_source,
            "play_audio": self._tool_play_audio,
            "create_study_plan": self._tool_create_study_plan,
        }

    # @MX:ANCHOR: 에이전트 공개 진입점 — api/routes/agent.py, __main__.py에서 호출
    # @MX:REASON: [AUTO] ReAct 루프 제어 흐름의 유일한 공개 진입점
    def run(self, question: str) -> AgentResult:
        """ReAct 루프 실행

        Args:
            question: 사용자 질문

        Returns:
            AgentResult (answer + steps 기록)
        """
        system_prompt = _REACT_SYSTEM_PROMPT.format(max_steps=self.max_steps)
        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        # 첫 user 메시지
        messages.append({"role": "user", "content": f"질문: {question}"})

        steps: list[AgentStep] = []

        for step_idx in range(self.max_steps):
            # ── Thought + Action 생성 ──
            raw = self.rag.provider.generate(
                messages=messages,
                model=self.rag.model,
                max_tokens=512,
            )
            logger.debug("step %d raw output: %s", step_idx, raw)

            # Final Answer 감지
            final_match = re.search(r"Final Answer[:：]\s*(.+)", raw, re.DOTALL)
            if final_match:
                answer = final_match.group(1).strip()
                return AgentResult(answer=answer, steps=steps, total_steps=step_idx + 1)

            # Thought / Action 파싱
            thought = _extract_thought(raw)
            action = _extract_action(raw)

            if action is None:
                # 파싱 실패 → 직접 답변으로 처리
                logger.warning("step %d: Action 파싱 실패, 응답을 Final Answer로 처리", step_idx)
                return AgentResult(answer=raw.strip(), steps=steps, total_steps=step_idx + 1)

            tool_name = action.get("tool", "")
            args = action.get("args", {})

            # ── Tool 실행 ──
            observation, retry_count = self._execute_with_retry(tool_name, args)

            step = AgentStep(
                thought=thought,
                tool=tool_name,
                args=args,
                observation=observation,
                retry_count=retry_count,
            )
            steps.append(step)

            # 메시지 히스토리에 assistant + observation 추가
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        # 스텝 초과 → 마지막 응답을 요약 요청
        messages.append({"role": "user", "content": "지금까지의 정보를 종합하여 최종 답변을 작성해 주세요."})
        final_raw = self.rag.provider.generate(
            messages=messages,
            model=self.rag.model,
            max_tokens=1024,
        )
        return AgentResult(answer=final_raw.strip(), steps=steps, total_steps=self.max_steps)

    def _execute_with_retry(
        self, tool_name: str, args: dict[str, Any]
    ) -> tuple[str, int]:
        """Tool 실행 + Self-Correction (설계서 12.4 시나리오 B)

        RRF 점수가 임계값 미만이면 쿼리를 재작성하여 재시도한다.
        """
        for retry in range(_MAX_QUERY_RETRY + 1):
            observation = self._call_tool(tool_name, args)

            # search 계열 tool에서 낮은 점수 감지 → 재시도
            if retry < _MAX_QUERY_RETRY and _is_low_score_result(observation):
                original_query = args.get("query") or args.get("word") or args.get("question", "")
                rewritten = self._rewrite_query(original_query)
                if rewritten and rewritten != original_query:
                    logger.info("Self-Correction: '%s' → '%s'", original_query, rewritten)
                    args = {**args, "query": rewritten}
                    continue
            return observation, retry

        return observation, _MAX_QUERY_RETRY

    def _call_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Tool 호출 및 결과를 문자열로 반환"""
        func = self._tools.get(tool_name)
        if func is None:
            return f"[오류] 존재하지 않는 도구: {tool_name}"
        try:
            return func(**args)
        except TypeError as e:
            return f"[오류] {tool_name} 인자 오류: {e}"
        except Exception as e:
            logger.exception("tool %s 실행 오류", tool_name)
            return f"[오류] {tool_name} 실행 실패: {e}"

    def _rewrite_query(self, query: str) -> str:
        """LLM으로 쿼리 재작성 (Self-Correction 보조)"""
        try:
            resp = self.rag.provider.generate(
                messages=[
                    {"role": "system", "content": "영어 학습 검색 쿼리 재작성 전문가입니다."},
                    {"role": "user", "content": f"'{query}'로 검색했으나 결과가 없었습니다. 더 효과적인 동의어나 관련어로 쿼리를 재작성해 주세요. 단어만 출력하세요."},
                ],
                model=self.rag.model,
                max_tokens=50,
            )
            return resp.strip().strip('"').strip("'")
        except Exception:
            return query

    # ───────────────────────────────────────
    # Tool 구현 (설계서 12.3)
    # ───────────────────────────────────────

    def _tool_search_word(
        self,
        query: str,
        top_k: int = 5,
        source: Optional[str] = None,
    ) -> str:
        """하이브리드 벡터 검색"""
        results: list[SearchResult] = self.retriever.search(
            query, top_k=top_k, source_filter=source
        )
        if not results:
            return "검색 결과 없음"
        lines = []
        for r in results:
            doc = r.document
            lines.append(f"[점수:{r.score:.3f}] {doc.word} — {doc.meaning} ({doc.source})")
        return "\n".join(lines)

    def _tool_rag_query(
        self,
        question: str,
        source: Optional[str] = None,
    ) -> str:
        """RAG 기반 자연어 응답 생성"""
        return self.rag.query(question=question, source_filter=source)

    def _tool_get_related_words(
        self,
        word: str,
        top_k: int = 5,
    ) -> str:
        """유의어·파생어 검색 (벡터 유사도 기반)"""
        results: list[SearchResult] = self.retriever.search(word, top_k=top_k + 1)
        # 자기 자신 제외
        related = [r for r in results if r.document.word.lower() != word.lower()][:top_k]
        if not related:
            return f"'{word}'의 유사 단어 없음"
        lines = [f"{r.document.word} — {r.document.meaning}" for r in related]
        return "\n".join(lines)

    def _tool_filter_by_source(
        self,
        source: str,
        query: str,
        top_k: int = 5,
    ) -> str:
        """특정 소스 필터 검색"""
        results: list[SearchResult] = self.retriever.search(
            query, top_k=top_k, source_filter=source
        )
        if not results:
            return f"source='{source}' 에서 '{query}' 결과 없음"
        lines = [f"{r.document.word} — {r.document.meaning}" for r in results]
        return "\n".join(lines)

    def _tool_play_audio(self, word: str) -> str:
        """발음 오디오 재생 (API 환경에서는 경로 반환)"""
        results = self.retriever.search(word, top_k=1)
        if results and results[0].document.audio_path:
            return f"오디오 경로: {results[0].document.audio_path}"
        return f"'{word}' 오디오 없음"

    def _tool_create_study_plan(self, topic: str, level: str = "intermediate") -> str:
        """학습 계획 수립 — LLM 생성"""
        prompt = (
            f"영어 학습 주제: {topic}\n"
            f"학습자 수준: {level}\n"
            f"위 조건으로 3단계 학습 계획을 간략히 작성해 주세요."
        )
        return self.rag.provider.generate(
            messages=[
                {"role": "system", "content": "영어 학습 계획 전문가입니다."},
                {"role": "user", "content": prompt},
            ],
            model=self.rag.model,
            max_tokens=300,
        )


# ───────────────────────────────────────────
# 파싱 헬퍼
# ───────────────────────────────────────────

def _extract_thought(text: str) -> str:
    m = re.search(r"Thought[:：]\s*(.+?)(?=\nAction|$)", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_action(text: str) -> Optional[dict]:
    m = re.search(r"Action[:：]\s*(\{.+?\})", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        # 단따옴표 → 쌍따옴표 교정 후 재시도
        fixed = m.group(1).replace("'", '"')
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            logger.warning("Action JSON 파싱 실패: %s", m.group(1))
            return None


def _is_low_score_result(observation: str) -> bool:
    """검색 결과 없음 또는 낮은 점수 감지"""
    return "검색 결과 없음" in observation or "결과 없음" in observation

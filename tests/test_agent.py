"""LearningAgent 테스트 — Self-RAG, Corrective RAG, ReAct 루프"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call

from src.agent import (
    LearningAgent,
    AgentResult,
    AgentStep,
    _extract_thought,
    _extract_action,
    _is_low_score_result,
    _is_search_tool,
)


# ---------------------------------------------------------------------------
# 공통 픽스처
# ---------------------------------------------------------------------------

def _make_agent(generate_side_effect=None, generate_return="Final Answer: 테스트 답변", max_steps=3):
    """LearningAgent 목 인스턴스 생성"""
    retriever = MagicMock()
    rag = MagicMock()
    rag.model = "test-model"

    if generate_side_effect is not None:
        rag.provider.generate.side_effect = generate_side_effect
    else:
        rag.provider.generate.return_value = generate_return

    return LearningAgent(retriever=retriever, rag=rag, max_steps=max_steps)


def _make_search_result(word="abandon", meaning="버리다", score=0.9, rank=1):
    """검색 결과 mock 생성 헬퍼"""
    from src.models import Document, SearchResult
    doc = Document(word=word, meaning=meaning, source="toefl", deck="TOEFL")
    return SearchResult(document=doc, score=score, rank=rank)


# ---------------------------------------------------------------------------
# TestNeedsRetrieval — Self-RAG 검색 필요 여부 판단
# ---------------------------------------------------------------------------
class TestNeedsRetrieval:

    def test_returns_true_when_llm_says_yes(self):
        """Given LLM이 YES를 반환할 때,
        When _needs_retrieval을 호출하면,
        Then True를 반환한다"""
        agent = _make_agent(generate_return="YES")
        assert agent._needs_retrieval("abandon 뜻이 뭐야?") is True

    def test_returns_false_when_llm_says_no(self):
        """Given LLM이 NO를 반환할 때,
        When _needs_retrieval을 호출하면,
        Then False를 반환한다"""
        agent = _make_agent(generate_return="NO")
        assert agent._needs_retrieval("안녕하세요") is False

    def test_returns_true_on_llm_error(self):
        """Given LLM 호출이 예외를 발생시킬 때,
        When _needs_retrieval을 호출하면,
        Then 안전 기본값 True를 반환한다"""
        agent = _make_agent(generate_side_effect=RuntimeError("API 오류"))
        assert agent._needs_retrieval("테스트") is True

    def test_case_insensitive_yes(self):
        """Given LLM이 소문자 yes를 반환할 때,
        When _needs_retrieval을 호출하면,
        Then True를 반환한다"""
        agent = _make_agent(generate_return="yes, 검색이 필요합니다.")
        assert agent._needs_retrieval("give up 의미") is True


# ---------------------------------------------------------------------------
# TestIsRelevantResult — Corrective RAG 관련성 평가
# ---------------------------------------------------------------------------
class TestIsRelevantResult:

    def test_returns_true_when_llm_says_yes(self):
        """Given LLM이 YES를 반환할 때,
        When _is_relevant_result를 호출하면,
        Then True를 반환한다"""
        agent = _make_agent(generate_return="YES")
        assert agent._is_relevant_result("abandon 뜻", "[점수:0.85] abandon — 버리다") is True

    def test_returns_false_when_llm_says_no(self):
        """Given LLM이 NO를 반환할 때,
        When _is_relevant_result를 호출하면,
        Then False를 반환한다"""
        agent = _make_agent(generate_return="NO, 관련 없음")
        assert agent._is_relevant_result("abandon 뜻", "persist — 지속하다") is False

    def test_returns_false_for_empty_observation(self):
        """Given observation이 빈 문자열일 때,
        When _is_relevant_result를 호출하면,
        Then False를 반환한다"""
        agent = _make_agent()
        assert agent._is_relevant_result("질문", "") is False

    def test_returns_false_when_no_results_in_observation(self):
        """Given observation이 '검색 결과 없음'을 포함할 때,
        When _is_relevant_result를 호출하면,
        Then LLM 호출 없이 False를 반환한다"""
        agent = _make_agent()
        agent.rag.provider.generate.assert_not_called()
        result = agent._is_relevant_result("abandon", "검색 결과 없음")
        assert result is False
        agent.rag.provider.generate.assert_not_called()

    def test_returns_true_on_llm_error(self):
        """Given LLM 호출이 예외를 발생시킬 때,
        When _is_relevant_result를 호출하면,
        Then 안전 기본값 True를 반환한다"""
        agent = _make_agent(generate_side_effect=RuntimeError("오류"))
        assert agent._is_relevant_result("질문", "어떤 검색 결과") is True


# ---------------------------------------------------------------------------
# TestRunSelfRAG — run()에서 Self-RAG 분기 검증
# ---------------------------------------------------------------------------
class TestRunSelfRAG:

    def test_no_retrieval_returns_direct_answer(self):
        """Given LLM이 검색 불필요(NO)로 판단할 때,
        When run을 호출하면,
        Then ReAct 루프 없이 LLM 직접 응답을 반환한다"""
        agent = _make_agent()
        # 첫 번째 generate 호출: Self-RAG → NO
        # 두 번째 generate 호출: 직접 응답
        agent.rag.provider.generate.side_effect = ["NO", "안녕하세요! 무엇을 도와드릴까요?"]

        result = agent.run("안녕하세요")

        assert isinstance(result, AgentResult)
        assert result.answer == "안녕하세요! 무엇을 도와드릴까요?"
        assert result.steps == []
        assert result.total_steps == 0

    def test_retrieval_needed_enters_react_loop(self):
        """Given LLM이 검색 필요(YES)로 판단할 때,
        When run을 호출하면,
        Then ReAct 루프로 진입한다"""
        agent = _make_agent()
        agent.rag.provider.generate.side_effect = [
            "YES",  # Self-RAG 판단
            "Final Answer: abandon은 '버리다' 입니다.",  # ReAct 최종 답변
        ]

        result = agent.run("abandon 뜻이 뭐야?")

        assert "버리다" in result.answer


# ---------------------------------------------------------------------------
# TestRunReActLoop — ReAct 루프 정상 동작
# ---------------------------------------------------------------------------
class TestRunReActLoop:

    def test_returns_final_answer_from_react(self):
        """Given ReAct 루프에서 Final Answer가 생성될 때,
        When run을 호출하면,
        Then 해당 답변을 포함한 AgentResult를 반환한다"""
        agent = _make_agent()
        react_output = (
            "Thought: 단어를 검색해야 한다\n"
            'Action: {"tool": "search_word", "args": {"query": "abandon"}}\n'
        )
        # 검색 결과가 있어야 Self-Correction이 안 걸리고 Corrective RAG로 진입
        agent.retriever.search.return_value = [_make_search_result()]
        agent.rag.provider.generate.side_effect = [
            "YES",  # Self-RAG: 검색 필요
            react_output,  # step 1: Action
            "YES",  # Corrective RAG: 관련성 충분
            "Final Answer: abandon은 '버리다'입니다.",  # step 2: 최종 답변
        ]

        result = agent.run("abandon 뜻")

        assert "abandon" in result.answer
        assert result.total_steps == 2

    def test_action_parse_failure_returns_raw_as_answer(self):
        """Given Action JSON 파싱이 실패할 때,
        When run을 호출하면,
        Then 해당 raw 응답을 Final Answer로 처리한다"""
        agent = _make_agent()
        agent.rag.provider.generate.side_effect = [
            "YES",  # Self-RAG
            "이것은 JSON이 아닌 일반 텍스트 응답입니다.",
        ]

        result = agent.run("테스트 질문")

        assert result.answer == "이것은 JSON이 아닌 일반 텍스트 응답입니다."

    def test_max_steps_exceeded_requests_summary(self):
        """Given 최대 스텝을 초과했을 때,
        When run을 호출하면,
        Then 요약 요청 후 최종 답변을 반환한다"""
        agent = _make_agent(max_steps=2)

        react_output = (
            "Thought: 검색 중\n"
            'Action: {"tool": "search_word", "args": {"query": "test"}}\n'
        )
        # 검색 결과 존재 + Corrective RAG YES → Self-Correction 안 걸림
        agent.retriever.search.return_value = [_make_search_result()]
        agent.rag.provider.generate.side_effect = [
            "YES",  # Self-RAG
            react_output,  # step 1 Action
            "YES",        # Corrective RAG step 1
            react_output,  # step 2 Action (max_steps 도달)
            "YES",        # Corrective RAG step 2
            "지금까지 수집한 정보를 종합한 답변입니다.",  # 요약 응답
        ]

        result = agent.run("복잡한 질문")

        assert result.total_steps == 2
        assert result.answer == "지금까지 수집한 정보를 종합한 답변입니다."

    def test_step_is_recorded_correctly(self):
        """Given Tool이 정상 실행될 때,
        When run을 호출하면,
        Then AgentStep에 thought/tool/args/observation이 기록된다"""
        agent = _make_agent()
        agent.retriever.search.return_value = [_make_search_result()]

        react_output = (
            "Thought: abandon을 검색한다\n"
            'Action: {"tool": "search_word", "args": {"query": "abandon"}}\n'
        )
        agent.rag.provider.generate.side_effect = [
            "YES",  # Self-RAG
            react_output,
            "YES",  # Corrective RAG 관련성 평가
            "Final Answer: abandon — 버리다",
        ]

        result = agent.run("abandon 뜻")

        assert len(result.steps) >= 1
        step = result.steps[0]
        assert step.tool == "search_word"
        assert step.args == {"query": "abandon"}
        assert "abandon" in step.thought


# ---------------------------------------------------------------------------
# TestExecuteWithRetry — Self-Correction + Corrective RAG
# ---------------------------------------------------------------------------
class TestExecuteWithRetry:

    def test_no_retry_when_result_relevant(self):
        """Given 검색 결과가 관련성 있을 때,
        When _execute_with_retry를 호출하면,
        Then 재시도 없이 observation을 반환한다"""
        agent = _make_agent(generate_return="YES")  # 관련성 평가 → YES
        # 검색 결과 존재 → Self-Correction 안 걸림
        agent.retriever.search.return_value = [_make_search_result()]

        obs, retry = agent._execute_with_retry(
            "search_word", {"query": "abandon"}, original_question="abandon 뜻"
        )
        assert retry == 0

    def test_self_correction_retries_on_no_results(self):
        """Given 검색 결과가 없을 때,
        When _execute_with_retry를 호출하면,
        Then 쿼리 재작성 후 재시도한다"""
        agent = _make_agent()
        # 쿼리 재작성 응답
        agent.rag.provider.generate.return_value = "abandon"
        agent.retriever.search.return_value = []

        obs, retry = agent._execute_with_retry(
            "search_word", {"query": "존재하지않는단어"}, original_question="없는단어 찾기"
        )
        assert retry > 0

    def test_corrective_rag_retries_on_irrelevant_result(self):
        """Given 검색 결과가 존재하지만 LLM이 NO(관련없음)로 평가할 때,
        When _execute_with_retry를 호출하면,
        Then 쿼리 재작성 후 재시도한다"""
        agent = _make_agent()
        agent.retriever.search.return_value = [_make_search_result(word="persist", meaning="지속하다")]

        # 첫 관련성 평가 → NO, 재작성 쿼리(원본과 다름) → YES
        agent.rag.provider.generate.side_effect = ["NO", "give up", "YES"]

        obs, retry = agent._execute_with_retry(
            "search_word", {"query": "abandon"}, original_question="abandon 뜻"
        )
        assert retry == 1

    def test_max_retry_is_respected(self):
        """Given 최대 재시도 횟수에 도달했을 때,
        When _execute_with_retry를 호출하면,
        Then 더 이상 재시도하지 않고 결과를 반환한다"""
        agent = _make_agent(generate_return="NO")  # 항상 관련없음
        agent.rag.provider.generate.side_effect = ["NO", "rewritten", "NO", "rewritten2", "NO"]
        agent.retriever.search.return_value = []

        obs, retry = agent._execute_with_retry(
            "search_word", {"query": "test"}, original_question="test 뜻"
        )
        # 최대 재시도 횟수 이하
        assert retry <= 2


# ---------------------------------------------------------------------------
# TestHelpers — 모듈 수준 헬퍼 함수
# ---------------------------------------------------------------------------
class TestHelpers:

    def test_extract_thought_normal(self):
        """Given Thought: 패턴이 있을 때,
        When _extract_thought를 호출하면,
        Then 추론 텍스트를 반환한다"""
        text = "Thought: 단어를 검색해야 한다\nAction: ..."
        assert _extract_thought(text) == "단어를 검색해야 한다"

    def test_extract_thought_missing(self):
        """Given Thought가 없을 때,
        When _extract_thought를 호출하면,
        Then 빈 문자열을 반환한다"""
        assert _extract_thought("이건 그냥 텍스트") == ""

    def test_extract_action_valid_json(self):
        """Given 유효한 JSON Action이 있을 때,
        When _extract_action을 호출하면,
        Then dict를 반환한다"""
        text = 'Action: {"tool": "search_word", "args": {"query": "test"}}'
        result = _extract_action(text)
        assert result == {"tool": "search_word", "args": {"query": "test"}}

    def test_extract_action_single_quotes_corrected(self):
        """Given 단따옴표 JSON Action이 있을 때,
        When _extract_action을 호출하면,
        Then 교정 후 dict를 반환한다"""
        text = "Action: {'tool': 'search_word', 'args': {'query': 'test'}}"
        result = _extract_action(text)
        assert result == {"tool": "search_word", "args": {"query": "test"}}

    def test_extract_action_invalid_returns_none(self):
        """Given 파싱 불가능한 JSON이 있을 때,
        When _extract_action을 호출하면,
        Then None을 반환한다"""
        text = "Action: {invalid json here}"
        assert _extract_action(text) is None

    def test_extract_action_missing_returns_none(self):
        """Given Action이 없을 때,
        When _extract_action을 호출하면,
        Then None을 반환한다"""
        assert _extract_action("Thought: 생각만 있음") is None

    def test_is_low_score_result_detects_no_results(self):
        """Given '검색 결과 없음' 텍스트가 있을 때,
        When _is_low_score_result를 호출하면,
        Then True를 반환한다"""
        assert _is_low_score_result("검색 결과 없음") is True

    def test_is_low_score_result_detects_result_없음(self):
        """Given '결과 없음' 텍스트가 있을 때,
        When _is_low_score_result를 호출하면,
        Then True를 반환한다"""
        assert _is_low_score_result("source='toefl' 에서 '테스트' 결과 없음") is True

    def test_is_low_score_result_returns_false_for_valid(self):
        """Given 정상 검색 결과가 있을 때,
        When _is_low_score_result를 호출하면,
        Then False를 반환한다"""
        assert _is_low_score_result("[점수:0.85] abandon — 버리다 (toefl)") is False

    def test_is_search_tool_returns_true_for_search_tools(self):
        """Given 검색 계열 Tool 이름일 때,
        When _is_search_tool을 호출하면,
        Then True를 반환한다"""
        for tool in ("search_word", "rag_query", "get_related_words", "filter_by_source"):
            assert _is_search_tool(tool) is True

    def test_is_search_tool_returns_false_for_non_search_tools(self):
        """Given 비검색 Tool 이름일 때,
        When _is_search_tool을 호출하면,
        Then False를 반환한다"""
        for tool in ("play_audio", "create_study_plan", "unknown_tool"):
            assert _is_search_tool(tool) is False

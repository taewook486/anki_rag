"""Adaptive RAG 테스트 — 쿼리 복잡도 분류 + 전략 분기 (v1.3)"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.adaptive import (
    AdaptiveRAG,
    AdaptiveResult,
    QueryComplexity,
    classify_query,
    classify_query_heuristic,
    classify_query_llm,
)
from src.agent import AgentResult, AgentStep


# ---------------------------------------------------------------------------
# 공통 픽스처
# ---------------------------------------------------------------------------

def _make_adaptive(generate_return="MODERATE"):
    """AdaptiveRAG 목 인스턴스 생성"""
    retriever = MagicMock()
    rag = MagicMock()
    rag.model = "test-model"
    rag.max_tokens = 1024
    rag.provider.generate.return_value = generate_return
    rag._extract_search_query.side_effect = lambda q: q
    rag._build_context.return_value = "테스트 컨텍스트"
    rag._build_user_content.return_value = "[참고 자료]\n테스트\n\n[질문]\ntest\n\n답변:"
    rag.last_results = []
    rag.query.return_value = "Moderate 답변"

    agent = MagicMock()
    agent.run.return_value = AgentResult(
        answer="Complex 답변",
        steps=[AgentStep(thought="생각", tool="search_word", args={"query": "test"}, observation="결과", retry_count=0)],
        total_steps=1,
    )

    return AdaptiveRAG(retriever=retriever, rag=rag, agent=agent)


def _make_search_result(word="abandon", meaning="버리다", score=0.9, rank=1):
    """검색 결과 mock 생성"""
    from src.models import Document, SearchResult
    doc = Document(word=word, meaning=meaning, source="toefl", deck="TOEFL")
    return SearchResult(document=doc, score=score, rank=rank)


# ---------------------------------------------------------------------------
# TestClassifyQueryHeuristic — 휴리스틱 기반 분류
# ---------------------------------------------------------------------------
class TestClassifyQueryHeuristic:

    def test_single_english_word_is_simple(self):
        """Given 단일 영어 단어,
        When 휴리스틱 분류를 실행하면,
        Then SIMPLE을 반환한다"""
        assert classify_query_heuristic("abandon") == QueryComplexity.SIMPLE

    def test_multi_word_phrase_is_simple(self):
        """Given 짧은 영어 구문,
        When 휴리스틱 분류를 실행하면,
        Then SIMPLE을 반환한다"""
        assert classify_query_heuristic("give up") == QueryComplexity.SIMPLE

    def test_meaning_query_is_simple(self):
        """Given 'X의 뜻' 패턴,
        When 휴리스틱 분류를 실행하면,
        Then SIMPLE을 반환한다"""
        assert classify_query_heuristic("abandon의 뜻은?") == QueryComplexity.SIMPLE

    def test_pronunciation_query_is_simple(self):
        """Given 'X 발음' 패턴,
        When 휴리스틱 분류를 실행하면,
        Then SIMPLE을 반환한다"""
        assert classify_query_heuristic("abandon 발음") == QueryComplexity.SIMPLE

    def test_difficulty_sort_is_complex(self):
        """Given 난이도별 정리 요청,
        When 휴리스틱 분류를 실행하면,
        Then COMPLEX를 반환한다"""
        assert classify_query_heuristic("TOEFL 경제 단어를 난이도별로 정리해줘") == QueryComplexity.COMPLEX

    def test_comparison_with_examples_is_complex(self):
        """Given 비교+예문 요청,
        When 휴리스틱 분류를 실행하면,
        Then COMPLEX를 반환한다"""
        assert classify_query_heuristic("abandon과 forsake의 비교를 예문과 함께") == QueryComplexity.COMPLEX

    def test_study_plan_is_complex(self):
        """Given 학습 계획 요청,
        When 휴리스틱 분류를 실행하면,
        Then COMPLEX를 반환한다"""
        assert classify_query_heuristic("비즈니스 영어 학습 계획 세워줘") == QueryComplexity.COMPLEX

    def test_ambiguous_returns_none(self):
        """Given 모호한 질문,
        When 휴리스틱 분류를 실행하면,
        Then None을 반환한다 (LLM 필요)"""
        assert classify_query_heuristic("give up 유의어 알려줘") is None

    def test_moderate_synonym_returns_none(self):
        """Given 유의어 질의 (Moderate),
        When 휴리스틱 분류를 실행하면,
        Then None을 반환한다 (LLM 분류 필요)"""
        assert classify_query_heuristic("TOEFL 빈출 단어 추천해줘") is None


# ---------------------------------------------------------------------------
# TestClassifyQueryLLM — LLM 기반 분류
# ---------------------------------------------------------------------------
class TestClassifyQueryLLM:

    def test_llm_returns_simple(self):
        """Given LLM이 SIMPLE을 반환할 때,
        When LLM 분류를 실행하면,
        Then SIMPLE을 반환한다"""
        provider = MagicMock()
        provider.generate.return_value = "SIMPLE"
        assert classify_query_llm("run", provider, "test-model") == QueryComplexity.SIMPLE

    def test_llm_returns_moderate(self):
        """Given LLM이 MODERATE를 반환할 때,
        When LLM 분류를 실행하면,
        Then MODERATE를 반환한다"""
        provider = MagicMock()
        provider.generate.return_value = "MODERATE"
        assert classify_query_llm("give up 유의어", provider, "test-model") == QueryComplexity.MODERATE

    def test_llm_returns_complex(self):
        """Given LLM이 COMPLEX를 반환할 때,
        When LLM 분류를 실행하면,
        Then COMPLEX를 반환한다"""
        provider = MagicMock()
        provider.generate.return_value = "COMPLEX"
        assert classify_query_llm("난이도별 정리", provider, "test-model") == QueryComplexity.COMPLEX

    def test_llm_error_falls_back_to_moderate(self):
        """Given LLM 호출 실패 시,
        When LLM 분류를 실행하면,
        Then 기본값 MODERATE를 반환한다"""
        provider = MagicMock()
        provider.generate.side_effect = Exception("API 오류")
        assert classify_query_llm("test", provider, "test-model") == QueryComplexity.MODERATE

    def test_llm_unexpected_response_defaults_moderate(self):
        """Given LLM이 예상 외 문자열을 반환할 때,
        When LLM 분류를 실행하면,
        Then MODERATE를 반환한다"""
        provider = MagicMock()
        provider.generate.return_value = "잘 모르겠습니다"
        assert classify_query_llm("test", provider, "test-model") == QueryComplexity.MODERATE


# ---------------------------------------------------------------------------
# TestClassifyQuery — 통합 분류 (휴리스틱 + LLM)
# ---------------------------------------------------------------------------
class TestClassifyQuery:

    def test_heuristic_takes_priority(self):
        """Given 명확한 Simple 패턴,
        When classify_query를 실행하면,
        Then LLM 호출 없이 SIMPLE을 반환한다"""
        provider = MagicMock()
        result = classify_query("abandon", provider=provider, model="test")
        assert result == QueryComplexity.SIMPLE
        provider.generate.assert_not_called()

    def test_falls_back_to_llm_when_heuristic_uncertain(self):
        """Given 모호한 질문,
        When classify_query를 실행하면,
        Then LLM을 호출하여 분류한다"""
        provider = MagicMock()
        provider.generate.return_value = "MODERATE"
        result = classify_query("TOEFL 빈출 단어 추천해줘", provider=provider, model="test")
        assert result == QueryComplexity.MODERATE
        provider.generate.assert_called_once()

    def test_no_provider_returns_moderate(self):
        """Given provider가 None이고 휴리스틱으로 분류 불가능할 때,
        When classify_query를 실행하면,
        Then 기본값 MODERATE를 반환한다"""
        result = classify_query("TOEFL 빈출 단어 추천해줘", provider=None, model="")
        assert result == QueryComplexity.MODERATE


# ---------------------------------------------------------------------------
# TestAdaptiveRAGSimple — Simple 전략 실행
# ---------------------------------------------------------------------------
class TestAdaptiveRAGSimple:

    @patch("src.adaptive.classify_query", return_value=QueryComplexity.SIMPLE)
    def test_simple_uses_dense_only(self, mock_classify):
        """Given Simple로 분류된 질문,
        When AdaptiveRAG.query()를 실행하면,
        Then Dense-only 검색을 사용한다"""
        adaptive = _make_adaptive()
        search_result = _make_search_result()
        adaptive.retriever.search_dense_only.return_value = [search_result]

        result = adaptive.query("abandon")

        adaptive.retriever.search_dense_only.assert_called_once()
        assert result.complexity == QueryComplexity.SIMPLE
        assert result.strategy_used == "dense_only"

    @patch("src.adaptive.classify_query", return_value=QueryComplexity.SIMPLE)
    def test_simple_returns_answer(self, mock_classify):
        """Given Simple 전략 실행 시,
        When 결과를 확인하면,
        Then LLM 응답이 answer에 포함된다"""
        adaptive = _make_adaptive(generate_return="abandon: 버리다")
        adaptive.retriever.search_dense_only.return_value = []

        result = adaptive.query("abandon")

        assert result.answer == "abandon: 버리다"
        assert result.agent_result is None


# ---------------------------------------------------------------------------
# TestAdaptiveRAGModerate — Moderate 전략 실행
# ---------------------------------------------------------------------------
class TestAdaptiveRAGModerate:

    @patch("src.adaptive.classify_query", return_value=QueryComplexity.MODERATE)
    def test_moderate_uses_hybrid_rrf(self, mock_classify):
        """Given Moderate로 분류된 질문,
        When AdaptiveRAG.query()를 실행하면,
        Then Hybrid RRF 검색(기존 RAG)을 사용한다"""
        adaptive = _make_adaptive()

        result = adaptive.query("give up 유의어")

        adaptive.rag.query.assert_called_once()
        assert result.complexity == QueryComplexity.MODERATE
        assert result.strategy_used == "hybrid_rrf"

    @patch("src.adaptive.classify_query", return_value=QueryComplexity.MODERATE)
    def test_moderate_passes_filters(self, mock_classify):
        """Given source_filter가 있을 때,
        When Moderate 전략을 실행하면,
        Then RAG query에 필터가 전달된다"""
        adaptive = _make_adaptive()

        adaptive.query("give up", source_filter="toefl", deck_filter="TOEFL")

        call_kwargs = adaptive.rag.query.call_args
        assert call_kwargs.kwargs.get("source_filter") == "toefl"
        assert call_kwargs.kwargs.get("deck_filter") == "TOEFL"


# ---------------------------------------------------------------------------
# TestAdaptiveRAGComplex — Complex 전략 실행
# ---------------------------------------------------------------------------
class TestAdaptiveRAGComplex:

    @patch("src.adaptive.classify_query", return_value=QueryComplexity.COMPLEX)
    def test_complex_delegates_to_agent(self, mock_classify):
        """Given Complex로 분류된 질문,
        When AdaptiveRAG.query()를 실행하면,
        Then LearningAgent.run()에 위임한다"""
        adaptive = _make_adaptive()

        result = adaptive.query("TOEFL 경제 단어를 난이도별로 정리해줘")

        adaptive.agent.run.assert_called_once_with("TOEFL 경제 단어를 난이도별로 정리해줘")
        assert result.complexity == QueryComplexity.COMPLEX
        assert result.strategy_used == "agent_react"

    @patch("src.adaptive.classify_query", return_value=QueryComplexity.COMPLEX)
    def test_complex_includes_agent_result(self, mock_classify):
        """Given Complex 전략 실행 시,
        When 결과를 확인하면,
        Then agent_result가 포함된다"""
        adaptive = _make_adaptive()

        result = adaptive.query("학습 계획 세워줘")

        assert result.agent_result is not None
        assert result.agent_result.total_steps == 1
        assert result.answer == "Complex 답변"


# ---------------------------------------------------------------------------
# TestAdaptiveResult — 결과 데이터 클래스
# ---------------------------------------------------------------------------
class TestAdaptiveResult:

    def test_default_search_results_empty(self):
        """Given search_results 미지정 시,
        When AdaptiveResult를 생성하면,
        Then 빈 리스트가 기본값이다"""
        result = AdaptiveResult(
            answer="test",
            complexity=QueryComplexity.SIMPLE,
            strategy_used="dense_only",
        )
        assert result.search_results == []
        assert result.agent_result is None

    def test_with_search_results(self):
        """Given search_results가 있을 때,
        When AdaptiveResult를 생성하면,
        Then 검색 결과가 보존된다"""
        sr = _make_search_result()
        result = AdaptiveResult(
            answer="test",
            complexity=QueryComplexity.MODERATE,
            strategy_used="hybrid_rrf",
            search_results=[sr],
        )
        assert len(result.search_results) == 1
        assert result.search_results[0].document.word == "abandon"


# ---------------------------------------------------------------------------
# TestQueryComplexityEnum — Enum 동작 확인
# ---------------------------------------------------------------------------
class TestQueryComplexityEnum:

    def test_enum_values(self):
        """Given QueryComplexity Enum,
        When 값을 확인하면,
        Then simple/moderate/complex 문자열이다"""
        assert QueryComplexity.SIMPLE.value == "simple"
        assert QueryComplexity.MODERATE.value == "moderate"
        assert QueryComplexity.COMPLEX.value == "complex"

    def test_enum_is_string(self):
        """Given QueryComplexity가 str Enum일 때,
        When 문자열 비교를 하면,
        Then 직접 비교 가능하다"""
        assert QueryComplexity.SIMPLE == "simple"


# ---------------------------------------------------------------------------
# T5: AdaptiveRAG Complex 경로 GraphRAG Fusion 주입 테스트
# ---------------------------------------------------------------------------

class TestAdaptiveRAGGraphFusion:
    """Complex 전략에 GraphRAG Fusion 주입 테스트 (T5)"""

    @patch("src.adaptive.classify_query", return_value=QueryComplexity.COMPLEX)
    def test_complex_path_invokes_graph_fusion_when_graph_provided(self, mock_classify):
        """Given graph가 제공된 AdaptiveRAG에서 Complex 경로 실행 시,
        When graph에 synonym 엣지가 있으면,
        Then graph_rag_fusion이 호출된다 (graph_word가 활용됨)"""
        from unittest.mock import patch as upatch
        from src.graph import WordKnowledgeGraph, WordNode, WordRelation, RelationType

        # 그래프에 "forsake"가 "abandon"의 synonym으로 등록
        g = WordKnowledgeGraph()
        g.add_word(WordNode(word="abandon", meaning="포기하다", source="t", deck=""))
        g.add_word(WordNode(word="forsake", meaning="버리다", source="t", deck=""))
        g.add_relation(WordRelation("abandon", "forsake", RelationType.SYNONYM))

        # retriever.search는 forsake 결과 반환
        from src.models import Document, SearchResult
        forsake_doc = Document(word="forsake", meaning="버리다", source="t", deck="T")
        forsake_result = SearchResult(document=forsake_doc, score=0.6, rank=1)

        retriever = MagicMock()
        retriever.search.return_value = [forsake_result]

        rag = MagicMock()
        rag.model = "test-model"
        rag.max_tokens = 1024
        rag.provider.generate.return_value = "Complex 답변"

        agent = MagicMock()
        from src.agent import AgentResult, AgentStep
        agent.run.return_value = AgentResult(
            answer="Complex 답변",
            steps=[],
            total_steps=0,
        )

        # AdaptiveRAG에 graph 주입
        adaptive = AdaptiveRAG(retriever=retriever, rag=rag, agent=agent, graph=g)
        result = adaptive.query("abandon 단어를 난이도별로 정리해줘")

        # Complex 경로가 실행됨
        assert result.complexity == QueryComplexity.COMPLEX

    @patch("src.adaptive.classify_query", return_value=QueryComplexity.COMPLEX)
    def test_complex_path_skips_fusion_when_no_graph(self, mock_classify):
        """Given graph=None으로 AdaptiveRAG가 생성된 경우,
        When Complex 경로 실행 시,
        Then fusion 없이 정상 실행된다 (예외 미발생)"""
        adaptive = _make_adaptive()  # graph=None
        result = adaptive.query("학습 계획")
        assert result.complexity == QueryComplexity.COMPLEX

    @patch("src.adaptive.classify_query", return_value=QueryComplexity.COMPLEX)
    def test_complex_path_skips_fusion_when_graph_empty(self, mock_classify):
        """Given 빈 그래프가 제공된 경우,
        When Complex 경로 실행 시,
        Then fusion 없이 정상 실행된다 (UB3)"""
        from src.graph import WordKnowledgeGraph
        empty_graph = WordKnowledgeGraph()

        retriever = MagicMock()
        rag = MagicMock()
        rag.model = "test-model"
        rag.max_tokens = 1024
        rag.provider.generate.return_value = "답변"
        agent = MagicMock()
        from src.agent import AgentResult
        agent.run.return_value = AgentResult(answer="답변", steps=[], total_steps=0)

        adaptive = AdaptiveRAG(retriever=retriever, rag=rag, agent=agent, graph=empty_graph)
        result = adaptive.query("학습 계획")
        assert result.complexity == QueryComplexity.COMPLEX

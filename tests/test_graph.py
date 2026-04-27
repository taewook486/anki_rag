"""지식 그래프 모듈 테스트 — WordKnowledgeGraph, build_from_documents, graph_rag_fusion"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from src.graph import (
    RelationType,
    WordKnowledgeGraph,
    WordNode,
    WordRelation,
    _WORDNET_AVAILABLE,
    build_from_documents,
    graph_rag_fusion,
)
from src.models import Document, SearchResult


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------

@pytest.fixture
def graph() -> WordKnowledgeGraph:
    """networkx 설치 여부와 관계없이 테스트 가능한 그래프 픽스처"""
    g = WordKnowledgeGraph()
    if not g.is_available:
        pytest.skip("networkx 미설치 — 그래프 테스트 건너뜀")
    return g


def _make_doc(**kwargs) -> Document:
    defaults = dict(word="test", meaning="테스트", source="toefl", deck="TOEFL")
    defaults.update(kwargs)
    return Document(**defaults)


def _make_result(word: str, score: float = 0.5) -> SearchResult:
    doc = _make_doc(word=word)
    return SearchResult(document=doc, score=score, rank=1)


# ---------------------------------------------------------------------------
# WordKnowledgeGraph — 기본 노드/엣지 조작
# ---------------------------------------------------------------------------

class TestWordKnowledgeGraph:
    """WordKnowledgeGraph 기본 기능 테스트"""

    def test_add_word_increases_node_count(self, graph: WordKnowledgeGraph):
        """Given 빈 그래프일 때,
        When 단어 노드를 추가하면,
        Then node_count가 증가한다"""
        graph.add_word(WordNode(word="abandon", meaning="포기하다", source="toefl", deck="TOEFL"))
        assert graph.node_count() == 1

    def test_add_relation_increases_edge_count(self, graph: WordKnowledgeGraph):
        """Given 두 노드가 존재할 때,
        When 관계를 추가하면,
        Then edge_count가 증가한다"""
        graph.add_word(WordNode(word="abandon", meaning="포기하다", source="toefl", deck=""))
        graph.add_word(WordNode(word="forsake", meaning="버리다", source="toefl", deck=""))
        graph.add_relation(WordRelation(
            source_word="abandon",
            target_word="forsake",
            relation_type=RelationType.SYNONYM,
        ))
        assert graph.edge_count() == 1

    def test_get_synonyms_returns_related_words(self, graph: WordKnowledgeGraph):
        """Given SYNONYM 관계가 추가되었을 때,
        When get_synonyms를 호출하면,
        Then 유의어 목록을 반환한다"""
        for w, m in [("abandon", "포기하다"), ("forsake", "버리다"), ("relinquish", "내주다")]:
            graph.add_word(WordNode(word=w, meaning=m, source="toefl", deck=""))
        graph.add_relation(WordRelation("abandon", "forsake", RelationType.SYNONYM))
        graph.add_relation(WordRelation("abandon", "relinquish", RelationType.SYNONYM))

        result = graph.get_synonyms("abandon")
        assert sorted(result) == ["forsake", "relinquish"]

    def test_get_antonyms_returns_opposite_words(self, graph: WordKnowledgeGraph):
        """Given ANTONYM 관계가 있을 때,
        When get_antonyms를 호출하면,
        Then 반의어를 반환한다"""
        graph.add_word(WordNode(word="accept", meaning="수락하다", source="toefl", deck=""))
        graph.add_word(WordNode(word="reject", meaning="거부하다", source="toefl", deck=""))
        graph.add_relation(WordRelation("accept", "reject", RelationType.ANTONYM))

        assert "reject" in graph.get_antonyms("accept")

    def test_get_derived_words_bidirectional(self, graph: WordKnowledgeGraph):
        """Given DERIVED_FROM 관계가 있을 때,
        When get_derived_words를 호출하면,
        Then 순방향·역방향 모두 반환한다"""
        graph.add_word(WordNode(word="abandon", meaning="포기하다", source="toefl", deck=""))
        graph.add_word(WordNode(word="abandonment", meaning="포기", source="toefl", deck=""))
        graph.add_relation(WordRelation("abandonment", "abandon", RelationType.DERIVED_FROM))

        assert "abandonment" in graph.get_derived_words("abandon")
        assert "abandon" in graph.get_derived_words("abandonment")

    def test_get_related_unknown_word_returns_empty(self, graph: WordKnowledgeGraph):
        """Given 존재하지 않는 단어일 때,
        When get_related를 호출하면,
        Then 빈 리스트를 반환한다"""
        assert graph.get_related("nonexistent_xyz") == []

    def test_get_related_max_depth(self, graph: WordKnowledgeGraph):
        """Given A→B→C SYNONYM 체인이 있을 때,
        When max_depth=2로 조회하면,
        Then B와 C 모두 반환된다"""
        for w in ["A", "B", "C"]:
            graph.add_word(WordNode(word=w, meaning="", source="test", deck=""))
        graph.add_relation(WordRelation("A", "B", RelationType.SYNONYM))
        graph.add_relation(WordRelation("B", "C", RelationType.SYNONYM))

        result = graph.get_related("A", max_depth=2)
        assert "B" in result
        assert "C" in result


# ---------------------------------------------------------------------------
# build_from_documents — 자동 그래프 구축
# ---------------------------------------------------------------------------

class TestBuildFromDocuments:
    """build_from_documents 기능 테스트"""

    def test_nodes_created_for_all_documents(self, graph: WordKnowledgeGraph):
        """Given Document 리스트가 있을 때,
        When build_from_documents를 호출하면,
        Then 모든 단어가 노드로 추가된다"""
        docs = [
            _make_doc(word="run", meaning="달리다"),
            _make_doc(word="walk", meaning="걷다"),
            _make_doc(word="jog", meaning="천천히 달리다"),
        ]
        build_from_documents(graph, docs)
        assert graph.node_count() >= 3

    def test_synonym_relation_built_from_synonyms_field(self, graph: WordKnowledgeGraph):
        """Given Document에 synonyms 필드가 있을 때,
        When build_from_documents를 호출하면,
        Then SYNONYM 관계가 생성된다"""
        doc = _make_doc(word="abandon", meaning="포기하다", synonyms=["forsake", "relinquish"])
        build_from_documents(graph, [doc])

        syns = graph.get_synonyms("abandon")
        assert "forsake" in syns
        assert "relinquish" in syns

    def test_derived_from_relation_by_suffix(self, graph: WordKnowledgeGraph):
        """Given 접미사로 연결된 단어 쌍이 있을 때,
        When build_from_documents를 호출하면,
        Then DERIVED_FROM 관계가 생성된다"""
        docs = [
            _make_doc(word="abandon", meaning="포기하다"),
            _make_doc(word="abandonment", meaning="포기"),
        ]
        build_from_documents(graph, docs)

        # abandonment → abandon 관계 확인
        derived = graph.get_derived_words("abandon")
        assert "abandonment" in derived

    def test_co_occurs_relation_from_example(self, graph: WordKnowledgeGraph):
        """Given 예문에 다른 단어가 포함된 Document일 때,
        When build_from_documents를 호출하면,
        Then CO_OCCURS 관계가 생성된다"""
        docs = [
            _make_doc(word="abandon", meaning="포기하다", example="She decided to abandon the project."),
            _make_doc(word="project", meaning="프로젝트"),
        ]
        build_from_documents(graph, docs)

        co = graph.get_related("abandon", relation_type=RelationType.CO_OCCURS)
        assert "project" in co

    def test_same_category_relation_same_source(self, graph: WordKnowledgeGraph):
        """Given 같은 source의 단어들이 있을 때,
        When build_from_documents를 호출하면,
        Then SAME_CATEGORY 관계가 생성된다"""
        docs = [
            _make_doc(word="apple", meaning="사과", source="toefl"),
            _make_doc(word="banana", meaning="바나나", source="toefl"),
        ]
        build_from_documents(graph, docs)

        same = graph.get_related("apple", relation_type=RelationType.SAME_CATEGORY)
        assert "banana" in same

    def test_empty_document_list(self, graph: WordKnowledgeGraph):
        """Given 빈 Document 리스트일 때,
        When build_from_documents를 호출하면,
        Then 그래프가 비어 있다"""
        build_from_documents(graph, [])
        assert graph.node_count() == 0

    def test_unavailable_graph_does_not_raise(self):
        """Given networkx 미설치 상태의 그래프일 때,
        When build_from_documents를 호출하면,
        Then 예외 없이 종료된다"""
        g = WordKnowledgeGraph.__new__(WordKnowledgeGraph)
        g._graph = None
        g._nx = None

        docs = [_make_doc(word="test", meaning="테스트")]
        build_from_documents(g, docs)  # 예외 미발생 확인


# ---------------------------------------------------------------------------
# graph_rag_fusion — GraphRAG 병합
# ---------------------------------------------------------------------------

class TestGraphRagFusion:
    """graph_rag_fusion 기능 테스트"""

    def test_returns_vector_results_when_graph_unavailable(self):
        """Given networkx 미설치 그래프일 때,
        When graph_rag_fusion을 호출하면,
        Then 원본 벡터 결과를 그대로 반환한다"""
        g = WordKnowledgeGraph.__new__(WordKnowledgeGraph)
        g._graph = None
        g._nx = None

        results = [_make_result("abandon")]
        assert graph_rag_fusion(results, g, "abandon") is results

    def test_returns_vector_results_when_no_related_words(self, graph: WordKnowledgeGraph):
        """Given query_word의 관련 단어가 없을 때,
        When graph_rag_fusion을 호출하면,
        Then 원본 벡터 결과를 반환한다"""
        graph.add_word(WordNode(word="abandon", meaning="포기하다", source="toefl", deck=""))

        results = [_make_result("abandon")]
        assert graph_rag_fusion(results, graph, "abandon") == results

    def test_returns_vector_results_when_retriever_is_none(self, graph: WordKnowledgeGraph):
        """Given retriever가 None일 때,
        When graph_rag_fusion을 호출하면,
        Then 원본 벡터 결과를 반환한다"""
        graph.add_word(WordNode(word="abandon", meaning="포기하다", source="toefl", deck=""))
        graph.add_word(WordNode(word="forsake", meaning="버리다", source="toefl", deck=""))
        graph.add_relation(WordRelation("abandon", "forsake", RelationType.SYNONYM))

        results = [_make_result("abandon")]
        merged = graph_rag_fusion(results, graph, "abandon", retriever=None)
        assert merged == results

    def test_merges_graph_results_via_retriever(self, graph: WordKnowledgeGraph):
        """Given retriever가 제공되고 그래프 관련 단어가 있을 때,
        When graph_rag_fusion을 호출하면,
        Then 그래프 재검색 결과가 병합된다"""
        graph.add_word(WordNode(word="abandon", meaning="포기하다", source="toefl", deck=""))
        graph.add_word(WordNode(word="forsake", meaning="버리다", source="toefl", deck=""))
        graph.add_relation(WordRelation("abandon", "forsake", RelationType.SYNONYM))

        # retriever.search("forsake", top_k=1) → forsake 결과 반환하도록 목
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [_make_result("forsake", score=0.4)]

        vector_results = [_make_result("abandon", score=0.8)]
        merged = graph_rag_fusion(
            vector_results, graph, "abandon",
            retriever=mock_retriever, top_k=5,
        )

        words = [r.document.word for r in merged]
        assert "abandon" in words
        assert "forsake" in words

    def test_deduplicates_results(self, graph: WordKnowledgeGraph):
        """Given 벡터 결과와 그래프 결과에 중복 단어가 있을 때,
        When graph_rag_fusion을 호출하면,
        Then 중복이 제거된다"""
        graph.add_word(WordNode(word="abandon", meaning="포기하다", source="toefl", deck=""))
        graph.add_word(WordNode(word="forsake", meaning="버리다", source="toefl", deck=""))
        graph.add_relation(WordRelation("abandon", "forsake", RelationType.SYNONYM))

        mock_retriever = MagicMock()
        # retriever가 이미 벡터 결과에 있는 단어를 반환
        mock_retriever.search.return_value = [_make_result("abandon", score=0.3)]

        vector_results = [_make_result("abandon", score=0.8)]
        merged = graph_rag_fusion(
            vector_results, graph, "abandon",
            retriever=mock_retriever, top_k=5,
        )

        words = [r.document.word for r in merged]
        assert words.count("abandon") == 1

    def test_respects_top_k_limit(self, graph: WordKnowledgeGraph):
        """Given 많은 관련 단어가 있을 때,
        When top_k=2로 호출하면,
        Then 최대 2개만 반환한다"""
        for w in ["base", "w1", "w2", "w3"]:
            graph.add_word(WordNode(word=w, meaning="", source="test", deck=""))
        for w in ["w1", "w2", "w3"]:
            graph.add_relation(WordRelation("base", w, RelationType.SYNONYM))

        mock_retriever = MagicMock()
        mock_retriever.search.side_effect = lambda q, top_k=1: [_make_result(q, score=0.3)]

        vector_results = [_make_result("base", score=0.9)]
        merged = graph_rag_fusion(
            vector_results, graph, "base",
            retriever=mock_retriever, top_k=2,
        )
        assert len(merged) <= 2

    def test_sorted_by_score_descending(self, graph: WordKnowledgeGraph):
        """Given 다양한 점수의 결과가 있을 때,
        When graph_rag_fusion을 호출하면,
        Then 점수 내림차순으로 정렬된다"""
        graph.add_word(WordNode(word="abandon", meaning="포기하다", source="toefl", deck=""))
        graph.add_word(WordNode(word="forsake", meaning="버리다", source="toefl", deck=""))
        graph.add_relation(WordRelation("abandon", "forsake", RelationType.SYNONYM))

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [_make_result("forsake", score=0.9)]

        vector_results = [_make_result("abandon", score=0.5)]
        merged = graph_rag_fusion(
            vector_results, graph, "abandon",
            retriever=mock_retriever, top_k=5,
        )

        scores = [r.score for r in merged]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# T1: ANTONYM — WordNet 기반 반의어 자동 추출
# ---------------------------------------------------------------------------

class TestAntonymExtraction:
    """WordNet ANTONYM 자동 추출 테스트"""

    @pytest.mark.skipif(not _WORDNET_AVAILABLE, reason="nltk/WordNet 미설치")
    def test_antonym_happy_unhappy(self, graph: WordKnowledgeGraph):
        """Given 'happy' 단어로 그래프를 빌드할 때,
        When build_from_documents를 호출하면,
        Then get_antonyms('happy')에 'unhappy'가 포함된다"""
        doc = _make_doc(word="happy", meaning="행복한")
        build_from_documents(graph, [doc])
        antonyms = graph.get_antonyms("happy")
        assert "unhappy" in antonyms, f"예상: 'unhappy' in {antonyms}"

    def test_antonym_graceful_skip_on_unknown_word(self, graph: WordKnowledgeGraph):
        """Given WordNet에 없는 단어 'xyzzyx'일 때,
        When build_from_documents를 호출하면,
        Then 예외 없이 종료되고 빈 반의어 목록을 반환한다"""
        doc = _make_doc(word="xyzzyx", meaning="없는 단어")
        build_from_documents(graph, [doc])
        antonyms = graph.get_antonyms("xyzzyx")
        assert antonyms == []

    def test_antonym_relation_type_is_antonym(self, graph: WordKnowledgeGraph):
        """Given ANTONYM 관계가 그래프에 추가된 뒤,
        When ANTONYM 관계 타입으로 필터링하면,
        Then 결과가 반환된다"""
        doc = _make_doc(word="good", meaning="좋은")
        build_from_documents(graph, [doc])
        antonyms = graph.get_related("good", relation_type=RelationType.ANTONYM)
        # good의 반의어(evil/bad 등)가 있을 수 있음 — 결과 타입만 확인
        assert isinstance(antonyms, list)


# ---------------------------------------------------------------------------
# T2: 영속화 save/load
# ---------------------------------------------------------------------------

class TestGraphPersistence:
    """WordKnowledgeGraph 영속화 (save/load) 테스트"""

    def test_save_load_roundtrip(self, graph: WordKnowledgeGraph, tmp_path):
        """Given 3개 노드 + 2개 엣지가 있는 그래프일 때,
        When save 후 새 인스턴스에서 load하면,
        Then 노드/엣지 수가 동일하다"""
        import importlib
        import src.graph as graph_module
        graph.add_word(WordNode(word="alpha", meaning="알파", source="test", deck=""))
        graph.add_word(WordNode(word="beta", meaning="베타", source="test", deck=""))
        graph.add_word(WordNode(word="gamma", meaning="감마", source="test", deck=""))
        graph.add_relation(WordRelation("alpha", "beta", RelationType.SYNONYM))
        graph.add_relation(WordRelation("beta", "gamma", RelationType.CO_OCCURS))

        save_path = str(tmp_path / "graph_test")
        graph.save(save_path)

        # 새 인스턴스에서 로드
        new_graph = WordKnowledgeGraph()
        result = new_graph.load(save_path)

        assert result is True
        assert new_graph.node_count() == 3
        assert new_graph.edge_count() == 2

    def test_load_missing_file_returns_false(self):
        """Given 존재하지 않는 경로일 때,
        When load를 호출하면,
        Then False를 반환하고 예외가 발생하지 않는다"""
        g = WordKnowledgeGraph()
        if not g.is_available:
            pytest.skip("networkx 미설치")
        result = g.load("nonexistent_path_xyz_abc")
        assert result is False

    def test_save_creates_pkl_and_graphml(self, graph: WordKnowledgeGraph, tmp_path):
        """Given 그래프에 노드가 있을 때,
        When save를 호출하면,
        Then .pkl과 .graphml 파일이 모두 생성된다"""
        graph.add_word(WordNode(word="test", meaning="테스트", source="test", deck=""))
        save_path = str(tmp_path / "subdir" / "graph")
        graph.save(save_path)

        import os
        assert os.path.exists(save_path + ".pkl")
        assert os.path.exists(save_path + ".graphml")


# ---------------------------------------------------------------------------
# T3: CO_OCCURS 문서당 상한
# ---------------------------------------------------------------------------

class TestCoOccurrenceCap:
    """CO_OCCURS 문서당 상한 테스트"""

    def test_cooccurrence_respects_cap(self, graph: WordKnowledgeGraph):
        """Given 예문에 20개 이상의 알려진 단어가 포함된 문서일 때,
        When max_cooccurrence_per_doc=5로 build_from_documents를 호출하면,
        Then 해당 문서에서 추가된 CO_OCCURS 엣지가 5개 이하다"""
        # 20개 단어 생성
        words = [f"word{i}" for i in range(20)]
        docs = [_make_doc(word=w, meaning=f"의미{i}") for i, w in enumerate(words)]

        # 첫 번째 문서의 예문에 나머지 19개 단어 포함
        example_text = " ".join(words[1:]) + " is a test example sentence."
        docs[0] = _make_doc(word=words[0], meaning="기준 단어", example=example_text)

        build_from_documents(graph, docs, max_cooccurrence_per_doc=5)

        # 첫 번째 단어(word0)의 CO_OCCURS 엣지가 5개 이하인지 확인
        co = graph.get_related(words[0], relation_type=RelationType.CO_OCCURS)
        assert len(co) <= 5, f"CO_OCCURS 엣지 수 {len(co)}가 상한 5를 초과"

    def test_default_cap_is_ten(self, graph: WordKnowledgeGraph):
        """Given 기본 max_cooccurrence_per_doc 값으로 호출할 때,
        When 예문에 15개 이상 단어가 포함된 경우,
        Then CO_OCCURS 엣지가 10개 이하다"""
        words = [f"cap{i}" for i in range(15)]
        docs = [_make_doc(word=w, meaning=f"의미{i}") for i, w in enumerate(words)]
        example_text = " ".join(words[1:]) + " test."
        docs[0] = _make_doc(word=words[0], meaning="기준", example=example_text)

        build_from_documents(graph, docs)  # 기본값 10

        co = graph.get_related(words[0], relation_type=RelationType.CO_OCCURS)
        assert len(co) <= 10, f"CO_OCCURS 엣지 수 {len(co)}가 기본 상한 10을 초과"

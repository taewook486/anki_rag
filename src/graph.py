"""지식 그래프 모듈 — 단어 간 의미 관계 관리 (v2.0 예정)

설계서 섹션 13.2 구현:
- 노드: Word, Meaning, Category
- 엣지: SYNONYM, ANTONYM, DERIVED_FROM, CO_OCCURS, SAME_CATEGORY
- 백엔드: NetworkX (로컬), 추후 Neo4j로 교체 가능
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.models import Document
    from src.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# 파생어 접미사 패턴 (DERIVED_FROM 관계 자동 추출)
_DERIVED_SUFFIXES: tuple[str, ...] = (
    "ment", "tion", "sion", "ation", "ness", "ity", "ance", "ence",
    "er", "or", "ist", "ism",
    "ing", "ed",
    "ly", "ful", "less", "able", "ible",
    "ize", "ise", "ify",
    "al", "ive", "ous",
)

# ───────────────────────────────────────────
# 그래프 엣지 관계 타입 (설계서 13.2)
# ───────────────────────────────────────────

class RelationType(str, Enum):
    SYNONYM = "SYNONYM"           # 유의어
    ANTONYM = "ANTONYM"           # 반의어
    DERIVED_FROM = "DERIVED_FROM" # 파생어 관계 (abandonment ← abandon)
    CO_OCCURS = "CO_OCCURS"       # 공기 관계 (예문 기반)
    SAME_CATEGORY = "SAME_CATEGORY"  # 동일 도메인 묶음


@dataclass
class WordNode:
    """단어 노드"""

    word: str
    meaning: str
    source: str
    deck: str = ""
    difficulty: Optional[str] = None


@dataclass
class WordRelation:
    """단어 간 관계 엣지"""

    source_word: str
    target_word: str
    relation_type: RelationType
    weight: float = 1.0  # 관계 강도 (co-occurs: 공기 빈도 등)


# ───────────────────────────────────────────
# WordKnowledgeGraph
# ───────────────────────────────────────────

class WordKnowledgeGraph:
    """영어 단어 지식 그래프

    설계서 13.2 — 그래프 데이터 흐름:
        AnkiParser 결과 → graph_builder → 그래프 저장
        → Qdrant 벡터 검색 결과와 GraphRAG Fusion

    현재 구현: NetworkX 인메모리 (v2.0에서 Neo4j로 교체 예정)
    """

    def __init__(self) -> None:
        try:
            import networkx as nx
            # MultiDiGraph: 같은 노드 쌍 간 복수 관계 타입 지원
            # (예: abandon↔project 에 CO_OCCURS + SAME_CATEGORY 동시 허용)
            self._graph = nx.MultiDiGraph()
            self._nx = nx
            logger.info("WordKnowledgeGraph: NetworkX 백엔드 초기화 완료")
        except ImportError:
            self._graph = None
            self._nx = None
            logger.warning("networkx 미설치 — WordKnowledgeGraph 비활성화. pip install networkx")

    @property
    def is_available(self) -> bool:
        return self._graph is not None

    def add_word(self, node: WordNode) -> None:
        """단어 노드 추가"""
        if not self.is_available:
            return
        self._graph.add_node(
            node.word,
            meaning=node.meaning,
            source=node.source,
            deck=node.deck,
            difficulty=node.difficulty,
        )

    def add_relation(self, relation: WordRelation) -> None:
        """관계 엣지 추가"""
        if not self.is_available:
            return
        self._graph.add_edge(
            relation.source_word,
            relation.target_word,
            relation_type=relation.relation_type.value,
            weight=relation.weight,
        )

    def get_related(
        self,
        word: str,
        relation_type: Optional[RelationType] = None,
        max_depth: int = 1,
    ) -> list[str]:
        """관련 단어 조회

        Args:
            word: 기준 단어
            relation_type: 필터할 관계 타입 (None이면 전체)
            max_depth: 그래프 탐색 깊이

        Returns:
            관련 단어 목록
        """
        if not self.is_available or word not in self._graph:
            return []

        related: set[str] = set()
        frontier = {word}

        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for node in frontier:
                for neighbor in self._graph.successors(node):
                    # MultiDiGraph: get_edge_data returns {edge_key: attr_dict}
                    edge_dict = self._graph.get_edge_data(node, neighbor) or {}
                    matched = relation_type is None or any(
                        attrs.get("relation_type") == relation_type.value
                        for attrs in edge_dict.values()
                    )
                    if matched and neighbor != word:
                        related.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier

        return sorted(related)

    def get_synonyms(self, word: str) -> list[str]:
        """유의어 조회"""
        return self.get_related(word, RelationType.SYNONYM)

    def get_antonyms(self, word: str) -> list[str]:
        """반의어 조회"""
        return self.get_related(word, RelationType.ANTONYM)

    def get_derived_words(self, word: str) -> list[str]:
        """파생어 조회 (역방향 포함)"""
        if not self.is_available or word not in self._graph:
            return []
        derived = set()
        for neighbor in self._graph.successors(word):
            # MultiDiGraph: get_edge_data returns {edge_key: attr_dict}
            edge_dict = self._graph.get_edge_data(word, neighbor) or {}
            if any(
                attrs.get("relation_type") == RelationType.DERIVED_FROM.value
                for attrs in edge_dict.values()
            ):
                derived.add(neighbor)
        for neighbor in self._graph.predecessors(word):
            edge_dict = self._graph.get_edge_data(neighbor, word) or {}
            if any(
                attrs.get("relation_type") == RelationType.DERIVED_FROM.value
                for attrs in edge_dict.values()
            ):
                derived.add(neighbor)
        return sorted(derived)

    def node_count(self) -> int:
        return self._graph.number_of_nodes() if self.is_available else 0

    def edge_count(self) -> int:
        return self._graph.number_of_edges() if self.is_available else 0


# ───────────────────────────────────────────
# 그래프 자동 구축 헬퍼 (설계서 13.2)
# ───────────────────────────────────────────

# @MX:ANCHOR: Document 리스트 → WordKnowledgeGraph 변환 진입점
# @MX:REASON: [AUTO] AnkiParser 결과를 그래프로 변환하는 유일한 공개 진입점 (indexer, cli에서 호출)
def build_from_documents(graph: WordKnowledgeGraph, documents: list[Document]) -> None:
    """Document 리스트로 지식 그래프를 자동 구축한다.

    설계서 13.2 — 그래프 데이터 흐름:
        AnkiParser 결과 → build_from_documents() → WordKnowledgeGraph

    처리하는 관계 타입:
        SYNONYM      — document.synonyms 필드
        DERIVED_FROM — 접미사 패턴 매칭 (ment/tion/er/ing/ed 등)
        CO_OCCURS    — document.example 예문에서 공기 단어 추출
        SAME_CATEGORY — 같은 source 내 단어 묶음

    Args:
        graph: 대상 WordKnowledgeGraph 인스턴스
        documents: AnkiParser 파싱 결과 Document 리스트
    """
    if not graph.is_available:
        logger.warning("build_from_documents: networkx 미설치 — 그래프 구축 건너뜀")
        return

    # ── 1. 모든 단어 노드 추가 ──
    for doc in documents:
        graph.add_word(WordNode(
            word=doc.word,
            meaning=doc.meaning,
            source=doc.source,
            deck=doc.deck or "",
            difficulty=doc.difficulty,
        ))

    # ── 2. SYNONYM: document.synonyms 필드 활용 ──
    for doc in documents:
        for syn in doc.synonyms:
            if not syn or syn.lower() == doc.word.lower():
                continue
            # 유의어 노드가 그래프에 없으면 추가
            if syn not in graph._graph:
                graph.add_word(WordNode(
                    word=syn,
                    meaning="",
                    source=doc.source,
                    deck=doc.deck or "",
                ))
            graph.add_relation(WordRelation(
                source_word=doc.word,
                target_word=syn,
                relation_type=RelationType.SYNONYM,
            ))

    # ── 3. DERIVED_FROM: 접미사 패턴으로 파생어 관계 추출 ──
    word_lower_map: dict[str, str] = {doc.word.lower(): doc.word for doc in documents}
    for doc in documents:
        base = doc.word.lower()
        for suffix in _DERIVED_SUFFIXES:
            if not base.endswith(suffix):
                continue
            stem = base[: -len(suffix)]
            # 최소 어근 길이 3자 이상, 어근이 그래프에 존재해야 함
            if len(stem) < 3:
                continue
            if stem in word_lower_map and stem != base:
                graph.add_relation(WordRelation(
                    source_word=doc.word,
                    target_word=word_lower_map[stem],
                    relation_type=RelationType.DERIVED_FROM,
                ))
                break  # 한 단어에 여러 접미사가 매칭되면 첫 번째만 사용

    # ── 4. CO_OCCURS: 예문에서 공기 관계 추출 ──
    for doc in documents:
        if not doc.example:
            continue
        example_words = set(re.findall(r"\b[a-z]{3,}\b", doc.example.lower()))
        for ew in example_words:
            if ew in word_lower_map and ew != doc.word.lower():
                graph.add_relation(WordRelation(
                    source_word=doc.word,
                    target_word=word_lower_map[ew],
                    relation_type=RelationType.CO_OCCURS,
                    weight=1.0,
                ))

    # ── 5. SAME_CATEGORY: 같은 source 내 단어 묶음 (최대 20개로 제한) ──
    source_groups: dict[str, list[str]] = {}
    for doc in documents:
        source_groups.setdefault(doc.source, []).append(doc.word)

    for words in source_groups.values():
        sample = words[:20]  # 과도한 엣지 생성 방지
        for i, w1 in enumerate(sample):
            for w2 in sample[i + 1:]:
                graph.add_relation(WordRelation(
                    source_word=w1,
                    target_word=w2,
                    relation_type=RelationType.SAME_CATEGORY,
                    weight=0.5,
                ))

    logger.info(
        "build_from_documents: 노드 %d개, 엣지 %d개 구축 완료",
        graph.node_count(),
        graph.edge_count(),
    )


# ───────────────────────────────────────────
# GraphRAG Fusion 헬퍼 (설계서 13.2)
# ───────────────────────────────────────────

def graph_rag_fusion(
    vector_results: list,
    graph: WordKnowledgeGraph,
    query_word: str,
    retriever: Optional[Any] = None,
    relation_type: Optional[RelationType] = None,
    top_k: int = 5,
) -> list:
    """벡터 검색 결과 + 지식 그래프 관련 단어 Qdrant 재검색 후 병합

    설계서 13.2 — GraphRAG Fusion:
        1. 그래프에서 query_word의 관련 단어 조회
        2. 각 관련 단어를 retriever로 Qdrant 재검색
        3. 벡터 결과와 그래프 결과 병합 (중복 제거 + 점수 내림차순 정렬)

    Args:
        vector_results: HybridRetriever.search() 결과 (SearchResult 리스트)
        graph: WordKnowledgeGraph 인스턴스
        query_word: 기준 단어
        retriever: HybridRetriever 인스턴스 (None이면 그래프 결과 미추가)
        relation_type: 추가할 그래프 관계 타입 (None이면 전체)
        top_k: 최종 반환할 결과 수

    Returns:
        병합된 SearchResult 리스트 (중복 제거, 점수 내림차순)
    """
    if not graph.is_available:
        return vector_results

    graph_words = graph.get_related(query_word, relation_type=relation_type)
    if not graph_words:
        return vector_results

    # retriever가 없으면 그래프 탐색 결과 로그만 기록하고 반환
    if retriever is None:
        logger.debug("GraphRAG: retriever 미제공 — 그래프 후보 %d개 미사용", len(graph_words))
        return vector_results

    # 기존 벡터 결과의 단어 집합 (중복 제거용)
    existing_words: set[str] = {r.document.word.lower() for r in vector_results}
    merged = list(vector_results)

    # 그래프 관련 단어를 Qdrant에서 재검색하여 병합
    for related_word in graph_words:
        if related_word.lower() in existing_words:
            continue
        try:
            extra = retriever.search(related_word, top_k=1)
            for r in extra:
                if r.document.word.lower() not in existing_words:
                    existing_words.add(r.document.word.lower())
                    merged.append(r)
        except Exception:
            logger.debug("GraphRAG: '%s' 재검색 실패 — 건너뜀", related_word)

    # 점수 내림차순 정렬 후 top_k 반환
    merged.sort(key=lambda r: r.score, reverse=True)
    logger.info(
        "GraphRAG Fusion: 벡터 %d개 + 그래프 추가 %d개 → 병합 %d개",
        len(vector_results),
        len(merged) - len(vector_results),
        min(len(merged), top_k),
    )
    return merged[:top_k]

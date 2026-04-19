"""GraphRAG API 라우트

SPEC-GRAPHRAG-001 E4, E5:
    GET /api/graph/related/{word}  — 관련 단어 조회
    GET /api/graph/stats           — 그래프 통계 조회
"""

import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.graph import RelationType, WordKnowledgeGraph

router = APIRouter()

# 전역 그래프 싱글톤 — None이면 lazy 초기화
# @MX:NOTE: [AUTO] 빌드 전 상태에서는 빈 그래프를 반환하여 S1 요구사항 준수
_graph: Optional[WordKnowledgeGraph] = None

_DEFAULT_GRAPH_PATH = "data/graph"


def get_graph() -> WordKnowledgeGraph:
    """WordKnowledgeGraph 인스턴스 반환 (lazy initialization)

    그래프 파일이 존재하면 로드, 없으면 빈 그래프 반환.
    """
    global _graph
    if _graph is None:
        graph_path = os.getenv("GRAPH_PATH", _DEFAULT_GRAPH_PATH)
        _graph = WordKnowledgeGraph(persist_path=graph_path)
    return _graph


# ---------------------------------------------------------------------------
# 응답 스키마
# ---------------------------------------------------------------------------

class RelatedWordItem(BaseModel):
    """관련 단어 아이템"""

    word: str
    relation_type: str


class GraphRelatedResponse(BaseModel):
    """GET /api/graph/related/{word} 응답"""

    word: str
    related: list[RelatedWordItem]


class GraphStatsResponse(BaseModel):
    """GET /api/graph/stats 응답"""

    node_count: int
    edge_count: int
    per_relation: dict[str, int]


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------

# @MX:ANCHOR: [AUTO] 그래프 관련 단어 조회 API 진입점
# @MX:REASON: [AUTO] Streamlit, 외부 클라이언트 등 복수 호출자 대상 공개 엔드포인트
@router.get("/graph/related/{word}", response_model=GraphRelatedResponse)
async def get_related_words(
    word: str,
    relation_type: Optional[RelationType] = Query(
        default=None,
        description="필터할 관계 타입 (SYNONYM/ANTONYM/DERIVED_FROM/CO_OCCURS/SAME_CATEGORY)",
    ),
) -> GraphRelatedResponse:
    """단어의 관련 단어 조회

    - **word**: 기준 단어
    - **relation_type**: 관계 타입 필터 (선택). 미지정 시 전체 관계 반환.
    """
    try:
        graph = get_graph()
        related_words = graph.get_related(word, relation_type=relation_type)

        # 관계 타입별로 다시 조회하여 각 단어의 relation_type 포함
        if relation_type is not None:
            items = [
                RelatedWordItem(word=w, relation_type=relation_type.value)
                for w in related_words
            ]
        else:
            # 모든 관계 타입을 순회하며 어떤 타입으로 연결되었는지 파악
            items = []
            seen: set[str] = set()
            for rt in RelationType:
                for w in graph.get_related(word, relation_type=rt):
                    if w not in seen:
                        items.append(RelatedWordItem(word=w, relation_type=rt.value))
                        seen.add(w)

        return GraphRelatedResponse(word=word, related=items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"그래프 조회 실패: {e}")


@router.get("/graph/stats", response_model=GraphStatsResponse)
async def get_graph_stats() -> GraphStatsResponse:
    """그래프 통계 조회

    - **node_count**: 전체 노드 수
    - **edge_count**: 전체 엣지 수
    - **per_relation**: 관계 타입별 엣지 수
    """
    try:
        graph = get_graph()

        # 관계 타입별 엣지 수 집계
        per_relation: dict[str, int] = {rt.value: 0 for rt in RelationType}
        if graph.is_available and graph._graph is not None:
            for _, _, data in graph._graph.edges(data=True):
                rt_val = data.get("relation_type", "")
                if rt_val in per_relation:
                    per_relation[rt_val] += 1

        return GraphStatsResponse(
            node_count=graph.node_count(),
            edge_count=graph.edge_count(),
            per_relation=per_relation,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"그래프 통계 조회 실패: {e}")

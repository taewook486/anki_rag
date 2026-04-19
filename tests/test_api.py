"""GraphRAG API 라우트 테스트 — /api/graph/related/{word}, /api/graph/stats (T6)"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# 공통 픽스처
# ---------------------------------------------------------------------------

def _make_mock_graph(node_count: int = 3, edge_count: int = 5):
    """테스트용 WordKnowledgeGraph 목 객체 생성"""
    from src.graph import RelationType

    mock = MagicMock()
    mock.is_available = True
    mock.node_count.return_value = node_count
    mock.edge_count.return_value = edge_count

    # 엣지별 카운트 목
    def _get_related(word, relation_type=None, max_depth=1):
        if relation_type == RelationType.SYNONYM:
            return ["forsake", "relinquish"]
        if relation_type == RelationType.ANTONYM:
            return ["accept"]
        return ["forsake", "relinquish", "accept"]

    mock.get_related.side_effect = _get_related

    # by_relation 통계 목
    mock._graph = MagicMock()
    mock._graph.edges.return_value = [
        (None, None, {"relation_type": "SYNONYM"}),
        (None, None, {"relation_type": "SYNONYM"}),
        (None, None, {"relation_type": "ANTONYM"}),
        (None, None, {"relation_type": "CO_OCCURS"}),
        (None, None, {"relation_type": "CO_OCCURS"}),
    ]

    return mock


# ---------------------------------------------------------------------------
# T6-A: GET /api/graph/related/{word}
# ---------------------------------------------------------------------------

class TestGraphRelatedRoute:
    """GET /api/graph/related/{word} 라우트 테스트"""

    def test_get_related_returns_200(self):
        """Given 그래프에 단어가 있을 때,
        When GET /api/graph/related/{word}를 호출하면,
        Then 200 응답과 JSON을 반환한다"""
        mock_graph = _make_mock_graph()

        from src.api.main import app
        import src.api.routes.graph as graph_route
        with patch.object(graph_route, "_graph", mock_graph):
            client = TestClient(app)
            response = client.get("/api/graph/related/abandon")

        assert response.status_code == 200
        data = response.json()
        assert "word" in data
        assert data["word"] == "abandon"
        assert "related" in data
        assert isinstance(data["related"], list)

    def test_get_related_with_relation_type_filter(self):
        """Given relation_type 파라미터를 지정할 때,
        When GET /api/graph/related/{word}?relation_type=SYNONYM을 호출하면,
        Then 해당 타입만 필터링된 결과를 반환한다"""
        mock_graph = _make_mock_graph()

        from src.api.main import app
        import src.api.routes.graph as graph_route
        with patch.object(graph_route, "_graph", mock_graph):
            client = TestClient(app)
            response = client.get("/api/graph/related/abandon?relation_type=SYNONYM")

        assert response.status_code == 200
        data = response.json()
        for item in data["related"]:
            assert item["relation_type"] == "SYNONYM"

    def test_get_related_invalid_relation_type_returns_422(self):
        """Given 잘못된 relation_type 파라미터일 때,
        When GET /api/graph/related/{word}?relation_type=INVALID를 호출하면,
        Then 422 응답을 반환한다"""
        mock_graph = _make_mock_graph()

        from src.api.main import app
        import src.api.routes.graph as graph_route
        with patch.object(graph_route, "_graph", mock_graph):
            client = TestClient(app)
            response = client.get("/api/graph/related/abandon?relation_type=INVALID")

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# T6-B: GET /api/graph/stats
# ---------------------------------------------------------------------------

class TestGraphStatsRoute:
    """GET /api/graph/stats 라우트 테스트"""

    def test_get_stats_returns_200(self):
        """Given 그래프가 초기화된 상태에서,
        When GET /api/graph/stats를 호출하면,
        Then 200 응답과 node_count, edge_count, per_relation을 반환한다"""
        mock_graph = _make_mock_graph()

        from src.api.main import app
        import src.api.routes.graph as graph_route
        with patch.object(graph_route, "_graph", mock_graph):
            client = TestClient(app)
            response = client.get("/api/graph/stats")

        assert response.status_code == 200
        data = response.json()
        assert "node_count" in data
        assert "edge_count" in data
        assert "per_relation" in data
        assert isinstance(data["node_count"], int)
        assert isinstance(data["edge_count"], int)
        assert isinstance(data["per_relation"], dict)

    def test_get_stats_by_relation_keys(self):
        """Given stats를 조회할 때,
        When per_relation을 확인하면,
        Then RelationType 이름들이 키로 포함된다"""
        mock_graph = _make_mock_graph()

        from src.api.main import app
        import src.api.routes.graph as graph_route
        with patch.object(graph_route, "_graph", mock_graph):
            client = TestClient(app)
            response = client.get("/api/graph/stats")

        data = response.json()
        by_rel = data["per_relation"]
        # RelationType 값 중 적어도 하나는 있어야 함
        valid_keys = {"SYNONYM", "ANTONYM", "DERIVED_FROM", "CO_OCCURS", "SAME_CATEGORY"}
        assert len(set(by_rel.keys()) & valid_keys) > 0


# ---------------------------------------------------------------------------
# T6-C: POST /api/adaptive use_graph 파라미터
# ---------------------------------------------------------------------------

class TestAdaptiveRouteUseGraph:
    """POST /api/adaptive use_graph 파라미터 테스트"""

    def test_adaptive_accepts_use_graph_field(self):
        """Given use_graph 필드를 포함한 요청일 때,
        When POST /api/adaptive를 호출하면,
        Then 500 이외의 응답이 반환된다 (422 아님)"""
        with patch("src.api.routes.adaptive.get_adaptive") as mock_get:
            from src.adaptive import AdaptiveResult, QueryComplexity
            mock_adaptive = MagicMock()
            mock_adaptive.query.return_value = AdaptiveResult(
                answer="테스트 답변",
                complexity=QueryComplexity.SIMPLE,
                strategy_used="dense_only",
                search_results=[],
            )
            mock_get.return_value = mock_adaptive

            from src.api.main import app
            client = TestClient(app)
            response = client.post(
                "/api/adaptive",
                json={"question": "abandon 뜻", "use_graph": True},
            )

        # 422 (파라미터 거부)가 아니면 성공
        assert response.status_code != 422

    def test_adaptive_use_graph_false_does_not_raise(self):
        """Given use_graph=False로 요청할 때,
        When POST /api/adaptive를 호출하면,
        Then 200 응답을 반환한다"""
        with patch("src.api.routes.adaptive.get_adaptive") as mock_get:
            from src.adaptive import AdaptiveResult, QueryComplexity
            mock_adaptive = MagicMock()
            mock_adaptive.query.return_value = AdaptiveResult(
                answer="테스트 답변",
                complexity=QueryComplexity.SIMPLE,
                strategy_used="dense_only",
                search_results=[],
            )
            mock_get.return_value = mock_adaptive

            from src.api.main import app
            client = TestClient(app)
            response = client.post(
                "/api/adaptive",
                json={"question": "abandon 뜻", "use_graph": False},
            )

        assert response.status_code == 200

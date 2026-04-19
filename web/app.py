"""Anki RAG Streamlit 웹 애플리케이션 (v2.0)

GraphRAG 탭이 추가된 인터랙티브 학습 도구.
"""

from __future__ import annotations

import os
import sys

# src 모듈 임포트를 위한 경로 추가
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

st.set_page_config(
    page_title="Anki RAG",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------------------------
# 전역 그래프 인스턴스 (세션별 캐시)
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_graph():
    """WordKnowledgeGraph를 로드한다 (앱 기동 시 1회).

    그래프 파일이 없으면 빈 그래프를 반환한다.
    """
    from src.graph import WordKnowledgeGraph

    graph_path = os.getenv("GRAPH_PATH", "data/graph")
    return WordKnowledgeGraph(persist_path=graph_path)


# ---------------------------------------------------------------------------
# 탭 렌더링 함수
# ---------------------------------------------------------------------------

def _render_search_tab() -> None:
    """검색 탭: 단어 검색 기본 기능"""
    st.header("단어 검색")
    st.info("검색 기능은 FastAPI 백엔드와 연동됩니다. 백엔드를 먼저 실행하세요.")

    api_url = st.text_input("API URL", value="http://localhost:8000")
    query = st.text_input("검색어 입력")

    if st.button("검색") and query:
        try:
            import requests
            resp = requests.post(
                f"{api_url}/api/search",
                json={"query": query, "top_k": 10},
                timeout=10,
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    st.dataframe(
                        [{"단어": r["word"], "뜻": r["meaning"], "점수": r["score"]} for r in results]
                    )
                else:
                    st.warning("검색 결과가 없습니다.")
            else:
                st.error(f"API 오류: {resp.status_code}")
        except Exception as e:
            st.error(f"연결 실패: {e}")


def _render_graph_tab() -> None:
    """지식 그래프 탭: 단어 관계 시각화 (T7)

    단어를 입력받아 관련 단어를 조회하고 테이블/그래프로 표시한다.
    """
    st.header("지식 그래프 탐색")

    graph = _load_graph()

    if not graph.is_available:
        st.error("NetworkX가 설치되지 않아 그래프를 사용할 수 없습니다.")
        return

    # 그래프 상태 표시
    col1, col2 = st.columns(2)
    col1.metric("노드 수", graph.node_count())
    col2.metric("엣지 수", graph.edge_count())

    if graph.node_count() == 0:
        st.warning("그래프가 비어 있습니다. 먼저 /api/index를 통해 인덱싱을 실행하세요.")

    st.divider()

    # 단어 입력 및 관계 타입 선택
    col_word, col_rel = st.columns([2, 1])
    with col_word:
        word = st.text_input("단어 입력", placeholder="예: abandon")
    with col_rel:
        from src.graph import RelationType
        rel_options = ["전체"] + [rt.value for rt in RelationType]
        selected_rel = st.selectbox("관계 타입", rel_options)

    if st.button("관련 단어 조회") and word:
        relation_type = None if selected_rel == "전체" else RelationType(selected_rel)
        related_words = graph.get_related(word, relation_type=relation_type)

        if not related_words:
            st.info(f"'{word}'에 대한 관련 단어가 없습니다.")
        else:
            # 관계 타입별로 결과 정리
            rows = []
            if relation_type is not None:
                rows = [{"단어": w, "관계 타입": selected_rel} for w in related_words]
            else:
                seen: set[str] = set()
                for rt in RelationType:
                    for w in graph.get_related(word, relation_type=rt):
                        if w not in seen:
                            rows.append({"단어": w, "관계 타입": rt.value})
                            seen.add(w)

            st.success(f"'{word}'의 관련 단어 {len(rows)}개")
            st.dataframe(rows, use_container_width=True)

            # 선택적: graphviz 시각화 (간단한 1-hop 그래프)
            _try_render_graphviz(word, rows)


def _try_render_graphviz(word: str, rows: list[dict]) -> None:
    """graphviz_chart를 사용한 간단한 네트워크 다이어그램 렌더링.

    오류 발생 시 조용히 건너뜀.
    """
    try:
        if not rows:
            return

        lines = ["digraph {", f'  "{word}" [shape=box, style=filled, fillcolor=lightblue]']
        # 최대 20개만 시각화 (가독성)
        for row in rows[:20]:
            rel_label = row["관계 타입"]
            target = row["단어"]
            lines.append(f'  "{word}" -> "{target}" [label="{rel_label}"]')
        lines.append("}")

        dot_source = "\n".join(lines)
        st.graphviz_chart(dot_source)
    except Exception:
        pass  # graphviz 미설치 등 오류 시 조용히 건너뜀


# ---------------------------------------------------------------------------
# 메인 레이아웃
# ---------------------------------------------------------------------------

def main() -> None:
    """Streamlit 앱 메인 진입점"""
    st.title("Anki RAG — 영어 학습 도우미")

    tab_search, tab_graph = st.tabs(["단어 검색", "지식 그래프"])

    with tab_search:
        _render_search_tab()

    with tab_graph:
        _render_graph_tab()


if __name__ == "__main__":
    main()

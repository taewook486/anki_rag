"""Streamlit 메인 애플리케이션"""

import streamlit as st
import requests

# 페이지 설정
st.set_page_config(
    page_title="Anki RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API 엔드포인트
API_BASE_URL = "http://127.0.0.1:8000"


def main():
    """메인 함수"""
    st.title("📚 Anki RAG - 영어 학습 도우미")
    st.sidebar.markdown("---")

    # 페이지 선택
    page = st.sidebar.radio(
        "페이지 선택",
        ["🔍 검색", "💬 채팅", "🕸️ 지식 그래프", "⚙️ 관리"],
    )

    if page == "🔍 검색":
        show_search_page()
    elif page == "💬 채팅":
        show_chat_page()
    elif page == "🕸️ 지식 그래프":
        show_graph_page()
    elif page == "⚙️ 관리":
        show_admin_page()


def show_search_page():
    """검색 페이지"""
    st.header("🔍 단어 검색")

    # 검색어 입력
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("검색어", placeholder="예: abandon, give up...")
    with col2:
        st.write("")
        st.write("")
        search_button = st.button("🔍 검색", use_container_width=True)

    # 필터
    with st.expander("필터 옵션"):
        sources = st.multiselect(
            "데이터 출처",
            ["toefl", "xfer", "hacker_toeic", "hacker_green", "phrasal", "sentences"],
            default=["toefl", "xfer", "hacker_toeic", "hacker_green", "phrasal", "sentences"],
        )
        top_k = st.slider("결과 수", 1, 50, 10)

    # 검색 실행
    if search_button and query:
        with st.spinner("검색 중..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/search",
                    json={"query": query, "top_k": top_k, "source_filter": None},
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()

                # 결과 표시
                if data["results"]:
                    st.success(f"{len(data['results'])}개의 결과를 찾았습니다.")

                    for i, result in enumerate(data["results"], 1):
                        with st.container():
                            cols = st.columns([3, 1, 1, 1])
                            with cols[0]:
                                st.subheader(f"{i}. {result['word']}")
                            with cols[1]:
                                if result.get("audio_available") and result.get("audio_paths"):
                                    import hashlib
                                    for ap in result["audio_paths"]:
                                        audio_id = hashlib.md5(ap.encode()).hexdigest()
                                        audio_url = f"{API_BASE_URL}/api/audio/{audio_id}"
                                        st.audio(audio_url, format="audio/mpeg")
                            with cols[2]:
                                st.metric("점수", f"{result['score']:.2f}")
                            with cols[3]:
                                st.caption(result["source"])

                            st.info(result["meaning"])
                            if result.get("pronunciation"):
                                st.caption(f"📢 {result['pronunciation']}")
                            if result.get("example"):
                                st.text(f"💬 {result['example']}")
                            if result.get("example_translation"):
                                st.caption(f"📝 {result['example_translation']}")

                            st.markdown("---")
                else:
                    st.warning("검색 결과가 없습니다.")

            except requests.exceptions.RequestException as e:
                st.error(f"API 연결 실패: {e}")
                st.info("FastAPI 서버가 실행 중인지 확인하세요 (http://127.0.0.1:8000)")


def show_chat_page():
    """채팅 페이지"""
    st.header("💬 RAG 채팅")

    # 채팅 기록 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 어시스턴트 응답
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/api/query",
                        json={"question": prompt, "top_k": 5, "source_filter": None},
                        timeout=120,
                    )
                    response.raise_for_status()
                    data = response.json()
                    answer = data["answer"]

                    st.markdown(answer)

                    # 출처 표시
                    if data.get("sources"):
                        with st.expander("📚 출처"):
                            for source in data["sources"]:
                                st.text(f"- {source['word']} ({source['source']} - {source['deck']})")

                    # 응답 저장
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except requests.exceptions.RequestException as e:
                    st.error(f"API 연결 실패: {e}")

    # 사이드바: 새로운 대화
    if st.sidebar.button("🔄 새 대화"):
        st.session_state.messages = []
        st.rerun()


_RELATION_STYLE = {
    "SYNONYM":      {"color": "#2ca02c", "label": "유의어",   "icon": "🟢"},
    "ANTONYM":      {"color": "#d62728", "label": "반의어",   "icon": "🔴"},
    "DERIVED_FROM": {"color": "#1f77b4", "label": "파생어",   "icon": "🔵"},
    "CO_OCCURS":    {"color": "#ff7f0e", "label": "공기 관계", "icon": "🟠"},
    "SAME_CATEGORY":{"color": "#9467bd", "label": "동일 카테고리", "icon": "🟣"},
}


def _draw_graph_plot(center_word: str, related: list[dict]):
    """중심 단어와 인접 노드를 Plotly로 그린다. 방사형 레이아웃."""
    import math

    import plotly.graph_objects as go

    if not related:
        return None

    # 방사형 배치: 중심 (0,0) + 인접 N개를 원주 위에 균등 분포
    n = len(related)
    positions: dict[str, tuple[float, float]] = {center_word: (0.0, 0.0)}
    for i, item in enumerate(related):
        theta = 2 * math.pi * i / n
        positions[item["word"]] = (math.cos(theta), math.sin(theta))

    # 엣지 (관계 타입별로 분리하여 색상 구분)
    edge_traces = []
    for rel_type, style in _RELATION_STYLE.items():
        xs, ys = [], []
        for item in related:
            if item["relation_type"] != rel_type:
                continue
            x0, y0 = positions[center_word]
            x1, y1 = positions[item["word"]]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        if xs:
            edge_traces.append(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=style["color"], width=2),
                hoverinfo="none",
                name=f"{style['icon']} {style['label']}",
            ))

    # 노드
    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    for word, (x, y) in positions.items():
        node_x.append(x)
        node_y.append(y)
        node_text.append(word)
        node_size.append(40 if word == center_word else 26)
        node_color.append("#ffd166" if word == center_word else "#a8dadc")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=14, color="#1a1a1a"),
        marker=dict(size=node_size, color=node_color,
                    line=dict(width=2, color="#333")),
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        showlegend=True,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=520,
        plot_bgcolor="#fafafa",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.02),
    )
    return fig


def show_graph_page():
    """지식 그래프 페이지 — WordNet/규칙 기반 단어 관계 시각화"""
    st.header("🕸️ 지식 그래프 (GraphRAG v2.0)")

    # 그래프 통계 (상단 카드)
    try:
        stats_resp = requests.get(f"{API_BASE_URL}/api/graph/stats", timeout=10)
        if stats_resp.status_code == 200:
            stats = stats_resp.json()
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("노드", f"{stats.get('node_count', 0):,}")
            c2.metric("엣지", f"{stats.get('edge_count', 0):,}")
            per = stats.get("per_relation", {})
            c3.metric("🟢 SYNONYM", f"{per.get('SYNONYM', 0):,}")
            c4.metric("🔴 ANTONYM", f"{per.get('ANTONYM', 0):,}")
            c5.metric("🔵 DERIVED", f"{per.get('DERIVED_FROM', 0):,}")
            c6.metric("🟠 CO_OCCUR", f"{per.get('CO_OCCURS', 0):,}")
        else:
            st.warning("그래프 통계를 불러올 수 없습니다.")
    except requests.exceptions.RequestException as e:
        st.error(f"API 연결 실패: {e}")
        return

    st.markdown("---")

    # 예시 버튼이 누른 단어를 위젯 인스턴스화 전에 처리 (Streamlit 제약)
    # session_state는 widget key가 만들어진 뒤에는 수정 불가하므로 별도 보조 키 사용
    default_word: str = st.session_state.pop("_graph_word_pending", "")
    auto_run: bool = st.session_state.pop("_graph_auto_run", False)

    # 쿼리 입력 (widget에 key를 부여하지 않아 외부 state와 충돌 방지)
    col_q, col_t, col_btn = st.columns([4, 2, 1])
    with col_q:
        word = st.text_input(
            "단어 입력",
            value=default_word,
            placeholder="예: abandon, terminate, give, accept...",
        )
    with col_t:
        rel_filter = st.selectbox(
            "관계 타입 필터",
            ["전체", "SYNONYM", "ANTONYM", "DERIVED_FROM", "CO_OCCURS", "SAME_CATEGORY"],
        )
    with col_btn:
        st.write("")
        st.write("")
        go_btn = st.button("🔍 조회", use_container_width=True) or auto_run

    # 빠른 예시 단어 — 클릭 시 pending key 설정 후 rerun, 다음 사이클에서 자동 조회
    st.caption("빠른 예시 (클릭하여 조회):")
    example_cols = st.columns(6)
    for i, ex in enumerate(["abandon", "terminate", "accept", "begin", "quibble", "rescind"]):
        if example_cols[i].button(ex, key=f"ex_{ex}"):
            st.session_state["_graph_word_pending"] = ex
            st.session_state["_graph_auto_run"] = True
            st.rerun()

    if not (go_btn and word):
        st.info("👆 단어를 입력하거나 위의 예시 버튼을 눌러 조회하세요.")
        return

    # API 호출
    params = {}
    if rel_filter != "전체":
        params["relation_type"] = rel_filter
    try:
        with st.spinner(f"그래프 조회 중: {word}"):
            resp = requests.get(
                f"{API_BASE_URL}/api/graph/related/{word}",
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"조회 실패: {e}")
        return

    related = data.get("related", [])
    if not related:
        st.warning(
            f"`{word}` 에 대한 {('전체' if rel_filter=='전체' else rel_filter)} 관계가 그래프에 없습니다. "
            f"빠른 예시 단어로 시도해 보세요."
        )
        return

    # 좌: 그래프 시각화 / 우: 관계 목록
    st.success(f"`{word}` 의 인접 단어 {len(related)}개를 찾았습니다.")
    col_viz, col_list = st.columns([3, 2])

    with col_viz:
        fig = _draw_graph_plot(word, related)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    with col_list:
        st.markdown("**관계 목록**")
        for item in related:
            style = _RELATION_STYLE.get(item["relation_type"], {})
            icon = style.get("icon", "")
            label = style.get("label", item["relation_type"])
            st.markdown(
                f"{icon} **{item['word']}** — _{label}_  \n"
                f"<span style='color:#666;font-size:0.85em'>({item['relation_type']})</span>",
                unsafe_allow_html=True,
            )


def _trigger_indexing(data_dir: str, recreate: bool) -> None:
    """인덱싱 시작 요청"""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/index",
            json={"data_dir": data_dir, "recreate": recreate},
            timeout=10,
        )
        if resp.status_code == 409:
            st.warning("이미 인덱싱 중입니다.")
        else:
            resp.raise_for_status()
            st.rerun()
    except requests.exceptions.RequestException as e:
        st.error(f"인덱싱 시작 실패: {e}")


def show_admin_page():
    """관리 페이지"""
    import time

    st.header("⚙️ 시스템 관리")

    st.subheader("API 상태")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ API 서버가 정상 작동 중입니다")
        else:
            st.error("❌ API 서버 오류")
    except requests.exceptions.RequestException:
        st.error("❌ API 서버에 연결할 수 없습니다")
        st.info("FastAPI 서버를 먼저 시작하세요:")
        st.code("uvicorn src.api.main:app --host 127.0.0.1 --port 8000", language="bash")
        return

    st.markdown("---")

    st.subheader("데이터 인덱싱")

    col1, col2 = st.columns([3, 1])
    with col1:
        data_dir = st.text_input("데이터 디렉토리", value="./data")
    with col2:
        st.write("")
        recreate = st.checkbox("컬렉션 재생성", value=True)

    try:
        status_resp = requests.get(f"{API_BASE_URL}/api/index/status", timeout=5)
        status = status_resp.json()
    except requests.exceptions.RequestException:
        status = {"status": "idle", "progress": 0.0, "message": "", "total": 0, "indexed": 0, "error": ""}

    current_status = status.get("status", "idle")

    if current_status == "running":
        st.info(f"인덱싱 중... {status.get('message', '')}")
        progress_val = float(status.get("progress", 0.0))
        st.progress(progress_val)
        total = status.get("total", 0)
        indexed = status.get("indexed", 0)
        if total > 0:
            st.caption(f"{indexed:,} / {total:,} 문서 처리됨")
        time.sleep(1)
        st.rerun()

    elif current_status == "done":
        st.success(f"✅ {status.get('message', '인덱싱 완료')}")
        st.progress(1.0)
        if st.button("새 인덱싱 시작"):
            _trigger_indexing(data_dir, recreate)

    elif current_status == "error":
        st.error(f"❌ 인덱싱 실패: {status.get('error', '')}")
        if st.button("다시 시도"):
            _trigger_indexing(data_dir, recreate)

    else:
        if st.button("인덱싱 시작", type="primary", use_container_width=True):
            _trigger_indexing(data_dir, recreate)

    st.markdown("---")

    st.subheader("캐시 관리")
    try:
        cache_resp = requests.get(f"{API_BASE_URL}/api/cache/stats", timeout=5)
        if cache_resp.status_code == 200:
            cache_data = cache_resp.json()

            col_s, col_p = st.columns(2)
            with col_s:
                s = cache_data.get("search_cache", {})
                st.markdown("**검색 캐시 (Level 1)**")
                st.metric("엔트리", s.get("total_entries", 0))
                st.metric("적중률", f"{s.get('hit_rate', 0) * 100:.1f}%")
                st.caption(f"Hit: {s.get('hit_count', 0)} / Miss: {s.get('miss_count', 0)}")
            with col_p:
                p = cache_data.get("pipeline_cache", {})
                st.markdown("**파이프라인 캐시 (Level 2)**")
                st.metric("엔트리", p.get("total_entries", 0))
                st.metric("적중률", f"{p.get('hit_rate', 0) * 100:.1f}%")
                st.caption(f"Hit: {p.get('hit_count', 0)} / Miss: {p.get('miss_count', 0)}")

            if st.button("캐시 초기화", type="secondary"):
                try:
                    clear_resp = requests.delete(f"{API_BASE_URL}/api/cache", timeout=5)
                    if clear_resp.status_code == 200:
                        result = clear_resp.json()
                        st.success(f"캐시 초기화 완료: {result.get('cleared_entries', 0)}건 제거")
                        st.rerun()
                    else:
                        st.error("캐시 초기화 실패")
                except requests.exceptions.RequestException as e:
                    st.error(f"캐시 초기화 실패: {e}")
    except requests.exceptions.RequestException:
        st.warning("캐시 통계를 불러올 수 없습니다")

    st.markdown("---")
    st.subheader("서버 정보")
    st.info(f"API 엔드포인트: {API_BASE_URL}")
    st.info("API 문서: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)")


if __name__ == "__main__":
    main()

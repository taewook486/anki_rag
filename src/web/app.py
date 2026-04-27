"""Streamlit 메인 애플리케이션"""

import uuid

import requests
import streamlit as st

from src.web.history import clear_history, get_recent, get_stats, init_db, save_query

# 페이지 설정
st.set_page_config(
    page_title="Anki RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API 엔드포인트
API_BASE_URL = "http://127.0.0.1:8000"

# DB 초기화 (앱 시작 시 1회) — REQ-UI-020
init_db()


def _ensure_session_id() -> str:
    """세션 ID 생성/조회"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    return st.session_state.session_id


def main():
    """메인 함수"""
    st.title("📚 Anki RAG - 영어 학습 도우미")
    st.sidebar.markdown("---")

    # 페이지 선택 — REQ-UI-001: 4번째 탭 추가
    page = st.sidebar.radio(
        "페이지 선택",
        ["🔍 검색", "💬 채팅", "⚙️ 관리", "🕸️ 지식 그래프"],
    )

    if page == "🔍 검색":
        show_search_page()
    elif page == "💬 채팅":
        show_chat_page()
    elif page == "⚙️ 관리":
        show_admin_page()
    elif page == "🕸️ 지식 그래프":
        show_graph_page()


# ---------------------------------------------------------------------------
# 검색 페이지
# ---------------------------------------------------------------------------

def show_search_page():
    """검색 페이지 — REQ-UI-010~015, REQ-UI-021~023"""
    st.header("🔍 단어 검색")

    # REQ-UI-022: 최근 검색 이력 사이드바
    _render_history_sidebar()

    # REQ-UI-010: Adaptive RAG 모드 토글
    adaptive_mode = st.toggle("🧠 Adaptive RAG 모드", value=False, key="adaptive_toggle")

    # 검색어 입력
    # REQ-UI-023: 이력 클릭 재검색 — preseed 값 사용
    preseed = st.session_state.pop("preseed_query", "")
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("검색어", value=preseed, placeholder="예: abandon, give up...")
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
        if adaptive_mode:
            _run_adaptive_search(query, top_k)
        else:
            _run_normal_search(query, top_k)


def _run_normal_search(query: str, top_k: int) -> None:
    """기존 하이브리드 검색"""
    with st.spinner("검색 중..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/search",
                json={"query": query, "top_k": top_k, "source_filter": None},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if results:
                st.success(f"{len(results)}개의 결과를 찾았습니다.")
                _render_search_results(results)
                save_query(query, len(results), strategy_used="search",
                           session_id=_ensure_session_id())
            else:
                st.warning("검색 결과가 없습니다.")

        except requests.exceptions.RequestException as e:
            st.error(f"API 연결 실패: {e}")
            st.info("FastAPI 서버가 실행 중인지 확인하세요 (http://127.0.0.1:8000)")


def _run_adaptive_search(query: str, top_k: int) -> None:
    """Adaptive RAG 검색 — REQ-UI-010~015"""
    with st.spinner("Adaptive RAG 분석 중..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/adaptive",
                json={"question": query, "use_graph": True},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            # REQ-UI-012: 복잡도 배지 표시
            complexity = data.get("complexity", "unknown")
            strategy = data.get("strategy_used", "unknown")
            st.info(f"복잡도: **{complexity.upper()}** | 전략: **{strategy}**")

            # 답변 표시
            st.markdown(f"**답변:** {data.get('answer', '')}")

            # REQ-UI-013: Agent 스텝 (Complex 전략)
            agent_steps = data.get("agent_steps")
            if agent_steps:
                with st.expander(f"🤖 Agent 추론 과정 ({len(agent_steps)}단계)"):
                    for i, step in enumerate(agent_steps, 1):
                        st.markdown(f"**Step {i}** — `{step.get('tool', '')}`")
                        st.caption(f"💭 {step.get('thought', '')}")
                        st.caption(f"📋 결과: {step.get('observation', '')[:200]}")
                        st.markdown("---")

            # REQ-UI-014: 그래프 융합 결과
            if data.get("graph_used") and data.get("graph_terms"):
                st.success(f"그래프에서 추가된 관련 단어: {', '.join(data['graph_terms'])}")

            # 출처
            sources = data.get("sources", [])
            if sources:
                with st.expander("📚 출처"):
                    for src in sources:
                        st.text(f"- {src['word']} ({src['source']} - {src['deck']})")

            save_query(query, len(sources), strategy_used=complexity,
                       session_id=_ensure_session_id())

        except requests.exceptions.RequestException as e:
            # REQ-UI-015: 오류 표시 (자동 폴백 없음)
            st.error(f"Adaptive RAG 연결 실패: {e}")
            st.info("토글을 해제하면 기존 검색 모드로 전환됩니다.")


def _render_search_results(results: list) -> None:
    """검색 결과 공통 렌더링"""
    for i, result in enumerate(results, 1):
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


def _render_history_sidebar() -> None:
    """최근 검색 이력 사이드바 — REQ-UI-022, REQ-UI-023"""
    recent = get_recent(limit=20)
    if not recent:
        return

    st.sidebar.markdown("### 최근 검색")
    for row in recent:
        label = f"{row['query']} ({row['result_count']}건)"
        if st.sidebar.button(label, key=f"hist_{row['id']}"):
            # REQ-UI-023: 클릭 시 검색어 자동 입력 후 재검색
            st.session_state["preseed_query"] = row["query"]
            st.rerun()
    st.sidebar.markdown("---")


# ---------------------------------------------------------------------------
# 채팅 페이지 (기존 유지)
# ---------------------------------------------------------------------------

def show_chat_page():
    """채팅 페이지"""
    st.header("💬 RAG 채팅")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("질문을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

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

                    if data.get("sources"):
                        with st.expander("📚 출처"):
                            for source in data["sources"]:
                                st.text(f"- {source['word']} ({source['source']} - {source['deck']})")

                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except requests.exceptions.RequestException as e:
                    st.error(f"API 연결 실패: {e}")

    if st.sidebar.button("🔄 새 대화"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# 지식 그래프 페이지 — REQ-UI-001~007
# ---------------------------------------------------------------------------

def show_graph_page():
    """지식 그래프 탭 — REQ-UI-001~007"""
    st.header("🕸️ 지식 그래프")

    # REQ-UI-002: 그래프 통계 표시
    try:
        stats_resp = requests.get(f"{API_BASE_URL}/api/graph/stats", timeout=10)
        stats_resp.raise_for_status()
        stats = stats_resp.json()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("노드 수 (단어)", stats["node_count"])
        with col2:
            st.metric("엣지 수 (관계)", stats["edge_count"])

        if stats["per_relation"]:
            with st.expander("관계 타입별 분포"):
                for rel_type, count in stats["per_relation"].items():
                    st.caption(f"**{rel_type}**: {count:,}개")

    except requests.exceptions.RequestException as e:
        # REQ-UI-006: API 오류 처리
        st.error(f"그래프 통계 조회 실패: {e}")
        st.info("FastAPI 서버 상태를 확인하세요.")
        return

    st.markdown("---")

    # REQ-UI-003: 단어 관련어 검색
    st.subheader("단어 관련어 검색")

    col_q, col_rel, col_btn = st.columns([3, 2, 1])
    with col_q:
        word_input = st.text_input("단어 입력", placeholder="예: abandon")
    with col_rel:
        # REQ-UI-004: 관계 타입 필터
        rel_options = ["전체", "SYNONYM", "ANTONYM", "DERIVED_FROM", "CO_OCCURS", "SAME_CATEGORY"]
        relation_filter = st.selectbox("관계 타입", rel_options)
    with col_btn:
        st.write("")
        st.write("")
        graph_search = st.button("검색", use_container_width=True)

    if graph_search and word_input:
        # REQ-UI-004: 관계 타입 파라미터 전달
        params: dict = {}
        if relation_filter != "전체":
            params["relation_type"] = relation_filter

        try:
            resp = requests.get(
                f"{API_BASE_URL}/api/graph/related/{word_input}",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            related = data.get("related", [])
            if related:
                st.success(f"'{word_input}'의 관련 단어 {len(related)}개")

                # 관계 타입별로 그룹화하여 표시
                grouped: dict[str, list[str]] = {}
                for item in related:
                    rt = item["relation_type"]
                    grouped.setdefault(rt, []).append(item["word"])

                for rt, words in sorted(grouped.items()):
                    st.markdown(f"**{rt}** ({len(words)}개)")
                    st.write(", ".join(words))
            else:
                st.info(f"'{word_input}'의 관련 단어가 없습니다.")
                st.caption("인덱싱 후 그래프가 구축되어야 결과가 표시됩니다.")

        except requests.exceptions.RequestException as e:
            # REQ-UI-006: 오류 처리
            st.error(f"그래프 조회 실패: {e}")
            st.info("FastAPI 서버 상태를 확인하세요.")


# ---------------------------------------------------------------------------
# 관리 페이지
# ---------------------------------------------------------------------------

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

    # REQ-UI-024~025: 학습 이력 관리
    st.subheader("학습 이력")
    stats = get_stats()
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("전체 검색 횟수", stats["total"])
    with col_b:
        if stats["top_words"]:
            st.markdown("**많이 검색한 단어 Top 5**")
            for item in stats["top_words"]:
                st.caption(f"- {item['query']}: {item['count']}회")

    if st.button("이력 초기화", type="secondary"):
        if st.session_state.get("confirm_clear"):
            deleted = clear_history()
            st.success(f"이력 초기화 완료: {deleted}건 삭제")
            st.session_state.pop("confirm_clear", None)
            st.rerun()
        else:
            st.session_state["confirm_clear"] = True
            st.warning("한 번 더 클릭하면 모든 이력이 삭제됩니다.")

    st.markdown("---")
    st.subheader("서버 정보")
    st.info(f"API 엔드포인트: {API_BASE_URL}")
    st.info("API 문서: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)")


if __name__ == "__main__":
    main()

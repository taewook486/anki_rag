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
    page = st.sidebar.radio("페이지 선택", ["🔍 검색", "💬 채팅", "⚙️ 관리"])

    if page == "🔍 검색":
        show_search_page()
    elif page == "💬 채팅":
        show_chat_page()
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
                    timeout=10,
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


def _trigger_indexing(data_dir: str, recreate: bool) -> None:
    """인덱싱 시작 요청"""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/index",
            json={"data_dir": data_dir, "recreate": recreate},
            timeout=5,
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

"""Streamlit UI 4탭 자동화 검증 스크립트

실행:
    .venv/Scripts/python.exe d:/project/anki_rag/test_playwright_ui.py

전제:
    - uvicorn 가동 중 (127.0.0.1:8000)
    - streamlit 가동 중 (127.0.0.1:8501)
    - BGE-M3 워밍업 완료
"""

import json
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright, expect, TimeoutError as PlaywrightTimeoutError

STREAMLIT_URL = "http://127.0.0.1:8501"
SHOT_DIR = Path("d:/project/anki_rag/_workspace/2026-06-06-ui")
SHOT_DIR.mkdir(parents=True, exist_ok=True)

results = {"tabs": {}, "errors": []}


def wait_streamlit_idle(page, timeout_ms=20000):
    """Streamlit '실행 중' 인디케이터 사라질 때까지 대기."""
    try:
        page.wait_for_function(
            """() => {
                const running = document.querySelector('[data-testid="stStatusWidget"]');
                if (!running) return true;
                const txt = running.textContent || '';
                return !(txt.includes('Running') || txt.includes('RUNNING'));
            }""",
            timeout=timeout_ms,
        )
    except PlaywrightTimeoutError:
        pass
    page.wait_for_timeout(800)


def click_sidebar_radio(page, label_keyword):
    """좌측 사이드바 페이지 라디오 버튼 클릭."""
    # Streamlit radio 옵션은 라벨로 매치 — 사이드바 안의 라디오 그룹에서만 매치
    sidebar = page.locator('[data-testid="stSidebar"]')
    sidebar.locator(f'label:has-text("{label_keyword}")').first.click()
    page.wait_for_timeout(1500)
    wait_streamlit_idle(page, timeout_ms=30000)
    page.wait_for_timeout(1500)


def screenshot(page, name):
    p = SHOT_DIR / f"{name}.png"
    page.screenshot(path=str(p), full_page=True)
    return str(p)


def test_search_tab(page):
    print("--- 검색 탭 테스트 ---")
    click_sidebar_radio(page, "검색")
    page.wait_for_timeout(1500)
    shot1 = screenshot(page, "01_search_initial")
    print(f"  초기 화면: {shot1}")

    # 검색어 입력
    input_box = page.locator('input[aria-label="검색어"]').first
    input_box.fill("abandon")
    page.wait_for_timeout(500)

    # 검색 버튼 클릭
    page.locator('button:has-text("검색")').first.click()
    # 결과 본문에 abandon 텍스트가 등장할 때까지 명시적 대기 (검색 중 스피너 우회)
    try:
        page.wait_for_function(
            """() => {
                const txt = document.body.innerText || '';
                if (txt.includes('검색 중')) return false;
                return txt.includes('단념') || txt.includes('포기') ||
                       txt.toLowerCase().includes('abandon');
            }""",
            timeout=60000,
        )
    except PlaywrightTimeoutError:
        pass
    wait_streamlit_idle(page, timeout_ms=10000)
    page.wait_for_timeout(2000)
    shot2 = screenshot(page, "02_search_result_abandon")
    print(f"  abandon 결과: {shot2}")

    # 결과 검증 — abandon 단어가 화면에 노출되어야
    body_text = page.locator("body").inner_text()
    has_abandon = "abandon" in body_text.lower()
    has_meaning = "단념" in body_text or "포기" in body_text or "meaning" in body_text.lower()

    results["tabs"]["search"] = {
        "status": "PASS" if has_abandon else "FAIL",
        "has_abandon": has_abandon,
        "has_meaning": has_meaning,
        "screenshots": [shot1, shot2],
    }
    print(f"  결과: {results['tabs']['search']['status']}")


def test_chat_tab(page):
    print("--- 채팅 탭 테스트 ---")
    click_sidebar_radio(page, "채팅")
    page.wait_for_timeout(1500)
    shot1 = screenshot(page, "03_chat_initial")
    print(f"  초기 화면: {shot1}")

    # Streamlit chat_input — data-testid="stChatInput" 안의 textarea
    chat_input = page.locator('[data-testid="stChatInputTextArea"]').first
    if chat_input.count() == 0:
        # fallback: chat input 컨테이너 안의 textarea
        chat_input = page.locator('[data-testid="stChatInput"] textarea').first
    if chat_input.count() == 0:
        # 최종 fallback: placeholder 매치
        chat_input = page.locator('textarea[placeholder*="질문을"]').first
    # 시나리오 A (Simple) — 빠른 Adaptive 응답으로 메타데이터 노출 검증
    chat_input.fill("abandon meaning")
    page.wait_for_timeout(500)
    page.keyboard.press("Enter")
    # 1차 — spinner 등장 + 사라짐
    try:
        page.wait_for_selector('[data-testid="stSpinner"]', state="visible", timeout=10000)
    except PlaywrightTimeoutError:
        pass
    try:
        page.wait_for_selector('[data-testid="stSpinner"]', state="hidden", timeout=120000)
    except PlaywrightTimeoutError:
        pass
    # 2차 — 응답 본문에 "복잡도" 캡션 또는 "단념/포기" 등장 대기
    try:
        page.wait_for_function(
            """() => {
                const txt = document.body.innerText || '';
                return txt.includes('complexity') || txt.includes('단념') || txt.includes('포기');
            }""",
            timeout=60000,
        )
    except PlaywrightTimeoutError:
        pass
    page.wait_for_timeout(4000)
    shot2 = screenshot(page, "04_chat_response")
    print(f"  응답: {shot2}")

    body_text = page.locator("body").inner_text()
    # Adaptive 메타데이터 노출 확인 + 응답에 단어 등장 확인
    has_complexity = "complexity" in body_text
    has_strategy = "strategy" in body_text and ("agent_react" in body_text or "hybrid_rrf" in body_text or "dense_only" in body_text)
    has_abandon = "abandon" in body_text.lower() and ("단념" in body_text or "포기" in body_text)

    results["tabs"]["chat"] = {
        "status": "PASS" if (has_complexity and has_strategy and has_abandon) else "FAIL",
        "has_complexity_meta": has_complexity,
        "has_strategy_meta": has_strategy,
        "has_abandon_response": has_abandon,
        "screenshots": [shot1, shot2],
    }
    print(f"  결과: {results['tabs']['chat']['status']}")


def test_graph_tab(page):
    print("--- 지식 그래프 탭 테스트 ---")
    click_sidebar_radio(page, "지식 그래프")
    page.wait_for_timeout(2000)
    shot1 = screenshot(page, "05_graph_initial")
    print(f"  초기 화면: {shot1}")

    # 메트릭 카드 확인
    body_text = page.locator("body").inner_text()
    has_metrics = "노드" in body_text and "엣지" in body_text and "SYNONYM" in body_text
    has_examples = "abandon" in body_text and "terminate" in body_text

    # abandon 예시 버튼 클릭 — Streamlit 빠른 예시 6개 버튼 중 abandon
    try:
        # 메인 콘텐츠 영역 안의 정확히 "abandon" 텍스트를 가진 버튼
        main = page.locator('[data-testid="stMain"]')
        if main.count() == 0:
            main = page.locator('[data-testid="stAppViewContainer"]')
        # text= 는 정확 매치
        main.locator('button').filter(has_text="abandon").first.click(timeout=15000)
        page.wait_for_timeout(2000)
        wait_streamlit_idle(page, timeout_ms=30000)
        page.wait_for_timeout(2000)
        shot2 = screenshot(page, "06_graph_abandon_result")
        print(f"  abandon 그래프: {shot2}")
        body_after = page.locator("body").inner_text()
        has_synonym = "유의어" in body_after or "SYNONYM" in body_after
        has_related = "desert" in body_after or "forsake" in body_after or "vacate" in body_after or "desolate" in body_after
    except Exception as e:
        shot2 = None
        has_synonym = False
        has_related = False
        results["errors"].append(f"graph_tab abandon click: {e}")

    results["tabs"]["graph"] = {
        "status": "PASS" if (has_metrics and has_examples and has_synonym and has_related) else "FAIL",
        "has_metrics": has_metrics,
        "has_examples": has_examples,
        "has_synonym": has_synonym,
        "has_related": has_related,
        "screenshots": [shot1, shot2] if shot2 else [shot1],
    }
    print(f"  결과: {results['tabs']['graph']['status']}")


def test_admin_tab(page):
    print("--- 관리 탭 테스트 ---")
    click_sidebar_radio(page, "관리")
    page.wait_for_timeout(2000)
    shot1 = screenshot(page, "07_admin_initial")
    print(f"  초기 화면: {shot1}")

    body_text = page.locator("body").inner_text()
    has_api_status = "API 서버" in body_text or "정상 작동" in body_text
    has_indexing = "인덱싱" in body_text
    has_cache = "캐시" in body_text or "Cache" in body_text or "엔트리" in body_text

    results["tabs"]["admin"] = {
        "status": "PASS" if (has_api_status and has_indexing and has_cache) else "FAIL",
        "has_api_status": has_api_status,
        "has_indexing": has_indexing,
        "has_cache": has_cache,
        "screenshots": [shot1],
    }
    print(f"  결과: {results['tabs']['admin']['status']}")


def main():
    print(f"Streamlit UI 자동화 시작: {STREAMLIT_URL}")
    print(f"스크린샷 저장 위치: {SHOT_DIR}")
    print()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()

        try:
            page.goto(STREAMLIT_URL, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(3000)
            wait_streamlit_idle(page)

            test_search_tab(page)
            test_chat_tab(page)
            test_graph_tab(page)
            test_admin_tab(page)
        except Exception as e:
            results["errors"].append(f"main: {type(e).__name__}: {e}")
            print(f"오류: {e}")
        finally:
            context.close()
            browser.close()

    # 결과 저장
    summary_path = SHOT_DIR / "ui_test_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"요약 저장: {summary_path}")

    # 종합 판정
    passed = sum(1 for t in results["tabs"].values() if t.get("status") == "PASS")
    total = len(results["tabs"])
    print(f"종합: {passed}/{total} 탭 PASS, 에러 {len(results['errors'])}건")

    if results["errors"]:
        print("에러 목록:")
        for e in results["errors"]:
            print(f"  - {e}")

    sys.exit(0 if (passed == total and not results["errors"]) else 1)


if __name__ == "__main__":
    main()

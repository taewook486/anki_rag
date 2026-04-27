"""학습 이력 SQLite 저장소 테스트 — src/web/history.py

REQ-UI-020 ~ REQ-UI-025 인수 기준 검증
"""

from __future__ import annotations

import pytest
from pathlib import Path

from src.web.history import (
    clear_history,
    get_recent,
    get_stats,
    init_db,
    save_query,
)


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """격리된 임시 DB 경로 픽스처"""
    return tmp_path / "test_history.db"


# ---------------------------------------------------------------------------
# REQ-UI-020: DB 초기화
# ---------------------------------------------------------------------------


class TestInitDb:
    def test_creates_db_file(self, tmp_db: Path) -> None:
        """init_db 호출 시 DB 파일이 생성된다"""
        assert not tmp_db.exists()
        init_db(tmp_db)
        assert tmp_db.exists()

    def test_idempotent(self, tmp_db: Path) -> None:
        """중복 호출해도 예외 없이 동작한다"""
        init_db(tmp_db)
        init_db(tmp_db)  # 두 번 호출해도 오류 없음
        assert tmp_db.exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """중간 디렉토리가 없어도 자동 생성한다"""
        db_path = tmp_path / "nested" / "dir" / "history.db"
        init_db(db_path)
        assert db_path.exists()


# ---------------------------------------------------------------------------
# REQ-UI-021: 검색 이력 저장
# ---------------------------------------------------------------------------


class TestSaveQuery:
    def test_save_basic(self, tmp_db: Path) -> None:
        """기본 필드로 저장하면 DB에 1건이 추가된다"""
        save_query("abandon", 5, db_path=tmp_db)
        rows = get_recent(db_path=tmp_db)
        assert len(rows) == 1
        assert rows[0]["query"] == "abandon"
        assert rows[0]["result_count"] == 5

    def test_save_with_strategy_and_session(self, tmp_db: Path) -> None:
        """strategy_used, session_id도 저장된다"""
        save_query("give up", 3, strategy_used="hybrid_rrf", session_id="sess-abc", db_path=tmp_db)
        rows = get_recent(db_path=tmp_db)
        assert rows[0]["strategy_used"] == "hybrid_rrf"
        assert rows[0]["session_id"] == "sess-abc"

    def test_save_optional_fields_none(self, tmp_db: Path) -> None:
        """선택 필드 None으로 저장해도 오류 없음"""
        save_query("run", 0, strategy_used=None, session_id=None, db_path=tmp_db)
        rows = get_recent(db_path=tmp_db)
        assert rows[0]["strategy_used"] is None
        assert rows[0]["session_id"] is None

    def test_timestamp_auto_set(self, tmp_db: Path) -> None:
        """timestamp가 자동으로 설정된다"""
        save_query("word", 1, db_path=tmp_db)
        rows = get_recent(db_path=tmp_db)
        assert rows[0]["timestamp"] is not None
        assert len(rows[0]["timestamp"]) > 0


# ---------------------------------------------------------------------------
# REQ-UI-022: 최근 검색 이력 조회
# ---------------------------------------------------------------------------


class TestGetRecent:
    def test_returns_in_reverse_order(self, tmp_db: Path) -> None:
        """최근 저장된 순서로 반환된다 (시간 역순)"""
        save_query("first", 1, db_path=tmp_db)
        save_query("second", 2, db_path=tmp_db)
        save_query("third", 3, db_path=tmp_db)
        rows = get_recent(db_path=tmp_db)
        assert rows[0]["query"] == "third"
        assert rows[1]["query"] == "second"
        assert rows[2]["query"] == "first"

    def test_limit_respected(self, tmp_db: Path) -> None:
        """limit 파라미터가 지켜진다"""
        for i in range(10):
            save_query(f"word{i}", i, db_path=tmp_db)
        rows = get_recent(limit=5, db_path=tmp_db)
        assert len(rows) == 5

    def test_empty_db_returns_empty_list(self, tmp_db: Path) -> None:
        """빈 DB에서 조회하면 빈 리스트 반환"""
        rows = get_recent(db_path=tmp_db)
        assert rows == []

    def test_returns_dict_with_expected_keys(self, tmp_db: Path) -> None:
        """반환 딕셔너리에 필수 키가 포함된다"""
        save_query("test", 1, db_path=tmp_db)
        row = get_recent(db_path=tmp_db)[0]
        for key in ("id", "timestamp", "query", "result_count", "strategy_used", "session_id"):
            assert key in row


# ---------------------------------------------------------------------------
# REQ-UI-024: 학습 이력 통계
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_total_count(self, tmp_db: Path) -> None:
        """total이 저장된 레코드 수와 일치한다"""
        for i in range(3):
            save_query(f"w{i}", 1, db_path=tmp_db)
        stats = get_stats(db_path=tmp_db)
        assert stats["total"] == 3

    def test_top_words(self, tmp_db: Path) -> None:
        """top_words에 가장 많이 검색된 단어가 포함된다"""
        save_query("abandon", 1, db_path=tmp_db)
        save_query("abandon", 2, db_path=tmp_db)
        save_query("run", 1, db_path=tmp_db)
        stats = get_stats(db_path=tmp_db)
        assert stats["top_words"][0]["query"] == "abandon"
        assert stats["top_words"][0]["count"] == 2

    def test_top_words_max_5(self, tmp_db: Path) -> None:
        """top_words는 최대 5개를 반환한다"""
        for i in range(10):
            save_query(f"unique_word_{i}", 1, db_path=tmp_db)
        stats = get_stats(db_path=tmp_db)
        assert len(stats["top_words"]) <= 5

    def test_empty_stats(self, tmp_db: Path) -> None:
        """빈 DB에서 통계를 조회하면 total=0"""
        stats = get_stats(db_path=tmp_db)
        assert stats["total"] == 0
        assert stats["top_words"] == []


# ---------------------------------------------------------------------------
# REQ-UI-025: 이력 초기화
# ---------------------------------------------------------------------------


class TestClearHistory:
    def test_clears_all_records(self, tmp_db: Path) -> None:
        """clear_history 호출 후 레코드가 모두 삭제된다"""
        save_query("a", 1, db_path=tmp_db)
        save_query("b", 2, db_path=tmp_db)
        clear_history(db_path=tmp_db)
        assert get_recent(db_path=tmp_db) == []

    def test_returns_deleted_count(self, tmp_db: Path) -> None:
        """삭제된 레코드 수를 반환한다"""
        save_query("a", 1, db_path=tmp_db)
        save_query("b", 2, db_path=tmp_db)
        count = clear_history(db_path=tmp_db)
        assert count == 2

    def test_clear_empty_db(self, tmp_db: Path) -> None:
        """빈 DB에서 초기화해도 오류 없음"""
        count = clear_history(db_path=tmp_db)
        assert count == 0

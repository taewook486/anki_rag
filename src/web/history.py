"""학습 이력 SQLite 저장소 — REQ-UI-020~025

스키마:
    learning_history(id, timestamp, query, result_count, strategy_used, session_id)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

# 기본 DB 경로
DEFAULT_DB_PATH = Path("data/learning_history.db")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS learning_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    query          TEXT    NOT NULL,
    result_count   INTEGER NOT NULL DEFAULT 0,
    strategy_used  TEXT,
    session_id     TEXT
);
"""


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """DB 및 테이블 초기화 — 앱 시작 시 호출"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()


def save_query(
    query: str,
    result_count: int,
    strategy_used: Optional[str] = None,
    session_id: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """검색 이력 저장 — REQ-UI-021"""
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO learning_history (query, result_count, strategy_used, session_id)"
            " VALUES (?, ?, ?, ?)",
            (query, result_count, strategy_used, session_id),
        )
        conn.commit()


def get_recent(limit: int = 20, db_path: Path = DEFAULT_DB_PATH) -> list[dict]:
    """최근 검색 이력 조회 (시간 역순) — REQ-UI-022

    Returns:
        id, timestamp, query, result_count, strategy_used, session_id 딕셔너리 리스트
    """
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM learning_history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_stats(db_path: Path = DEFAULT_DB_PATH) -> dict:
    """학습 이력 통계 — REQ-UI-024

    Returns:
        {total: int, top_words: [{query, count}]}
    """
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        total: int = conn.execute("SELECT COUNT(*) FROM learning_history").fetchone()[0]
        top_rows = conn.execute(
            "SELECT query, COUNT(*) AS cnt"
            " FROM learning_history"
            " GROUP BY query"
            " ORDER BY cnt DESC"
            " LIMIT 5"
        ).fetchall()
    return {
        "total": total,
        "top_words": [{"query": row[0], "count": row[1]} for row in top_rows],
    }


def clear_history(db_path: Path = DEFAULT_DB_PATH) -> int:
    """이력 전체 삭제 — REQ-UI-025

    Returns:
        삭제된 레코드 수
    """
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("DELETE FROM learning_history")
        conn.commit()
    return cursor.rowcount

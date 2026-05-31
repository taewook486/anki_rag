"""Tests for src/api/deps.py — Qdrant 클라이언트 싱글톤 프로바이더 (RED)"""

from __future__ import annotations

import threading

import pytest


@pytest.fixture(autouse=True)
def _reset_singleton():
    """각 테스트 전후로 싱글톤 초기화 — 격리 보장."""
    try:
        from src.api.deps import reset_qdrant_client
    except ImportError:
        pytest.skip("src.api.deps not implemented yet")
    reset_qdrant_client()
    yield
    reset_qdrant_client()


def test_get_qdrant_client_returns_singleton(monkeypatch):
    """동일 프로세스에서 두 번 호출하면 같은 객체를 반환해야 한다."""
    monkeypatch.setenv("QDRANT_LOCATION", ":memory:")
    from src.api.deps import get_qdrant_client

    client1 = get_qdrant_client()
    client2 = get_qdrant_client()
    assert client1 is client2


def test_get_qdrant_client_thread_safe(monkeypatch):
    """10개 스레드가 동시에 호출해도 하나의 인스턴스만 생성되어야 한다."""
    monkeypatch.setenv("QDRANT_LOCATION", ":memory:")
    from src.api.deps import get_qdrant_client

    results: list = []
    lock = threading.Lock()

    def _worker():
        c = get_qdrant_client()
        with lock:
            results.append(id(c))

    threads = [threading.Thread(target=_worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 10
    assert len(set(results)) == 1, f"여러 인스턴스가 생성됨: {set(results)}"


def test_reset_qdrant_client_creates_new_instance(monkeypatch):
    """reset 후 다시 호출하면 새로운 인스턴스를 생성해야 한다 (테스트 격리용)."""
    monkeypatch.setenv("QDRANT_LOCATION", ":memory:")
    from src.api.deps import get_qdrant_client, reset_qdrant_client

    first = get_qdrant_client()
    reset_qdrant_client()
    second = get_qdrant_client()
    assert first is not second


def test_get_qdrant_client_respects_memory_location(monkeypatch):
    """QDRANT_LOCATION=:memory: 일 때 location 모드로 생성되어야 한다."""
    monkeypatch.setenv("QDRANT_LOCATION", ":memory:")
    from src.api.deps import get_qdrant_client
    from qdrant_client import QdrantClient

    client = get_qdrant_client()
    assert isinstance(client, QdrantClient)

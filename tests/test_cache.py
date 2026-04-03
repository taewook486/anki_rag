"""QueryCache 단위 테스트"""

import threading
import time

import pytest

from src.cache import QueryCache, make_cache_key, clear_all_caches


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------
class TestMakeCacheKey:
    def test_same_args_produce_same_key(self):
        key1 = make_cache_key(query="abandon", top_k=5)
        key2 = make_cache_key(query="abandon", top_k=5)
        assert key1 == key2

    def test_different_args_produce_different_key(self):
        key1 = make_cache_key(query="abandon", top_k=5)
        key2 = make_cache_key(query="abandon", top_k=10)
        assert key1 != key2

    def test_none_values_excluded(self):
        key1 = make_cache_key(query="abandon", source_filter=None)
        key2 = make_cache_key(query="abandon")
        assert key1 == key2

    def test_list_order_independent(self):
        key1 = make_cache_key(exclude_sources=["a", "b"])
        key2 = make_cache_key(exclude_sources=["b", "a"])
        assert key1 == key2

    def test_key_is_hex_string(self):
        key = make_cache_key(query="test")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


# ---------------------------------------------------------------------------
# QueryCache 기본 동작
# ---------------------------------------------------------------------------
class TestQueryCacheBasic:
    def test_get_set(self):
        cache = QueryCache(max_entries=10, ttl_seconds=60, enabled=True)
        cache.set("k1", "v1")
        assert cache.get("k1") == "v1"

    def test_get_missing_returns_none(self):
        cache = QueryCache(max_entries=10, ttl_seconds=60, enabled=True)
        assert cache.get("missing") is None

    def test_overwrite_existing(self):
        cache = QueryCache(max_entries=10, ttl_seconds=60, enabled=True)
        cache.set("k1", "v1")
        cache.set("k1", "v2")
        assert cache.get("k1") == "v2"

    def test_clear(self):
        cache = QueryCache(max_entries=10, ttl_seconds=60, enabled=True)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cleared = cache.clear()
        assert cleared == 2
        assert cache.get("k1") is None
        assert cache.get("k2") is None


# ---------------------------------------------------------------------------
# LRU 퇴거
# ---------------------------------------------------------------------------
class TestQueryCacheLRU:
    def test_evicts_oldest_when_full(self):
        cache = QueryCache(max_entries=3, ttl_seconds=60, enabled=True)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")
        # k1이 가장 오래됨 → 4번째 삽입 시 k1 퇴거
        cache.set("k4", "v4")
        assert cache.get("k1") is None  # 퇴거됨
        assert cache.get("k2") == "v2"
        assert cache.get("k3") == "v3"
        assert cache.get("k4") == "v4"

    def test_access_refreshes_lru_order(self):
        cache = QueryCache(max_entries=3, ttl_seconds=60, enabled=True)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")
        # k1을 조회하면 LRU 순서가 갱신됨
        cache.get("k1")
        # 이제 k2가 가장 오래됨 → k4 삽입 시 k2 퇴거
        cache.set("k4", "v4")
        assert cache.get("k1") == "v1"  # 살아남음
        assert cache.get("k2") is None  # 퇴거됨

    def test_max_entries_not_exceeded(self):
        cache = QueryCache(max_entries=5, ttl_seconds=60, enabled=True)
        for i in range(20):
            cache.set(f"k{i}", f"v{i}")
        stats = cache.stats()
        assert stats["total_entries"] <= 5


# ---------------------------------------------------------------------------
# TTL 만료
# ---------------------------------------------------------------------------
class TestQueryCacheTTL:
    def test_expired_entry_returns_none(self):
        cache = QueryCache(max_entries=10, ttl_seconds=1, enabled=True)
        cache.set("k1", "v1")
        assert cache.get("k1") == "v1"
        time.sleep(1.1)
        assert cache.get("k1") is None

    def test_expired_counts_as_miss(self):
        cache = QueryCache(max_entries=10, ttl_seconds=1, enabled=True)
        cache.set("k1", "v1")
        cache.get("k1")  # hit
        time.sleep(1.1)
        cache.get("k1")  # miss (expired)
        stats = cache.stats()
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 1


# ---------------------------------------------------------------------------
# 비활성화 상태
# ---------------------------------------------------------------------------
class TestQueryCacheDisabled:
    def test_disabled_get_returns_none(self):
        cache = QueryCache(max_entries=10, ttl_seconds=60, enabled=False)
        cache.set("k1", "v1")
        assert cache.get("k1") is None

    def test_disabled_set_does_nothing(self):
        cache = QueryCache(max_entries=10, ttl_seconds=60, enabled=False)
        cache.set("k1", "v1")
        stats = cache.stats()
        assert stats["total_entries"] == 0

    def test_disabled_stats_zero(self):
        cache = QueryCache(max_entries=10, ttl_seconds=60, enabled=False)
        cache.set("k1", "v1")
        cache.get("k1")
        stats = cache.stats()
        assert stats["hit_count"] == 0
        assert stats["miss_count"] == 0


# ---------------------------------------------------------------------------
# 통계
# ---------------------------------------------------------------------------
class TestQueryCacheStats:
    def test_hit_miss_tracking(self):
        cache = QueryCache(max_entries=10, ttl_seconds=60, enabled=True)
        cache.set("k1", "v1")
        cache.get("k1")       # hit
        cache.get("k1")       # hit
        cache.get("missing")  # miss
        stats = cache.stats()
        assert stats["hit_count"] == 2
        assert stats["miss_count"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_stats_with_no_queries(self):
        cache = QueryCache(max_entries=10, ttl_seconds=60, enabled=True)
        stats = cache.stats()
        assert stats["hit_rate"] == 0.0
        assert stats["total_entries"] == 0

    def test_stats_fields(self):
        cache = QueryCache(max_entries=100, ttl_seconds=3600, enabled=True)
        stats = cache.stats()
        assert "hit_count" in stats
        assert "miss_count" in stats
        assert "hit_rate" in stats
        assert "total_entries" in stats
        assert "max_entries" in stats
        assert "ttl_seconds" in stats
        assert "enabled" in stats


# ---------------------------------------------------------------------------
# 스레드 안전성
# ---------------------------------------------------------------------------
class TestQueryCacheThreadSafety:
    def test_concurrent_read_write(self):
        cache = QueryCache(max_entries=100, ttl_seconds=60, enabled=True)
        errors: list[Exception] = []

        def writer(start: int):
            try:
                for i in range(100):
                    cache.set(f"key-{start}-{i}", f"val-{start}-{i}")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    cache.get("key-0-50")
            except Exception as e:
                errors.append(e)

        threads = []
        for t_id in range(5):
            threads.append(threading.Thread(target=writer, args=(t_id,)))
        for _ in range(5):
            threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = cache.stats()
        assert stats["total_entries"] <= 100

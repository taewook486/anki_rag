"""응답 캐시 모듈 — 인메모리 LRU 캐시 (스레드 안전)"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 환경변수 기반 설정
# ---------------------------------------------------------------------------
def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning("환경변수 %s 값이 올바르지 않습니다: %s (기본값 %d 사용)", key, val, default)
        return default


# ---------------------------------------------------------------------------
# 캐시 키 생성
# ---------------------------------------------------------------------------
def make_cache_key(**kwargs: Any) -> str:
    """직렬화 가능한 파라미터로 SHA256 캐시 키를 생성한다.

    Args:
        **kwargs: 캐시 키에 포함할 파라미터 (None 값은 제외)

    Returns:
        64자 hex 문자열
    """
    # None 값 제거, 정렬된 키로 안정적인 해시
    filtered = {k: v for k, v in kwargs.items() if v is not None}
    # list/set → 정렬된 tuple 변환
    for k, v in filtered.items():
        if isinstance(v, (list, set, frozenset)):
            filtered[k] = sorted(v) if v else []
    key_str = json.dumps(filtered, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# QueryCache
# ---------------------------------------------------------------------------
class QueryCache:
    """스레드 안전 인메모리 LRU 캐시

    - OrderedDict 기반 O(1) LRU 퇴거
    - TTL 기반 만료
    - 적중/미스 통계 추적
    """

    def __init__(
        self,
        max_entries: int | None = None,
        ttl_seconds: int | None = None,
        enabled: bool | None = None,
    ):
        self._max_entries = max_entries if max_entries is not None else _env_int("CACHE_MAX_ENTRIES", 1000)
        self._ttl_seconds = ttl_seconds if ttl_seconds is not None else _env_int("CACHE_TTL", 86400)
        self._enabled = enabled if enabled is not None else _env_bool("CACHE_ENABLED", True)
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()  # key -> (value, created_at)
        self._lock = threading.Lock()
        self._hit_count = 0
        self._miss_count = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값을 조회한다.

        - 비활성화 상태이면 항상 None
        - TTL 만료 엔트리는 제거 후 None
        - 적중 시 LRU 순서를 갱신 (move_to_end)
        """
        if not self._enabled:
            return None

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._miss_count += 1
                logger.debug("캐시 미스: %s", key[:16])
                return None

            value, created_at = entry
            if time.time() - created_at > self._ttl_seconds:
                # TTL 만료
                del self._cache[key]
                self._miss_count += 1
                logger.debug("캐시 TTL 만료: %s", key[:16])
                return None

            # 적중: LRU 순서 갱신
            self._cache.move_to_end(key)
            self._hit_count += 1
            logger.debug("캐시 적중: %s", key[:16])
            return value

    def set(self, key: str, value: Any) -> None:
        """캐시에 값을 저장한다.

        - 비활성화 상태이면 무시
        - max_entries 초과 시 가장 오래된 엔트리를 퇴거
        """
        if not self._enabled:
            return

        with self._lock:
            # 이미 있으면 갱신
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = (value, time.time())
                return

            # LRU 퇴거
            while len(self._cache) >= self._max_entries:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug("캐시 LRU 퇴거: %s", evicted_key[:16])

            self._cache[key] = (value, time.time())

    def clear(self) -> int:
        """모든 캐시 엔트리를 제거한다.

        Returns:
            제거된 엔트리 수
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info("캐시 전체 초기화: %d건 제거", count)
            return count

    def stats(self) -> dict:
        """캐시 통계를 반환한다."""
        with self._lock:
            total = self._hit_count + self._miss_count
            return {
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": round(self._hit_count / total, 4) if total > 0 else 0.0,
                "total_entries": len(self._cache),
                "max_entries": self._max_entries,
                "ttl_seconds": self._ttl_seconds,
                "enabled": self._enabled,
            }


# ---------------------------------------------------------------------------
# 글로벌 캐시 인스턴스 (싱글톤)
# ---------------------------------------------------------------------------
_search_cache: Optional[QueryCache] = None
_pipeline_cache: Optional[QueryCache] = None
_global_lock = threading.Lock()


def get_search_cache() -> QueryCache:
    """Level 1 검색 결과 캐시 인스턴스를 반환한다."""
    global _search_cache
    if _search_cache is None:
        with _global_lock:
            if _search_cache is None:
                _search_cache = QueryCache()
    return _search_cache


def get_pipeline_cache() -> QueryCache:
    """Level 2 파이프라인 응답 캐시 인스턴스를 반환한다."""
    global _pipeline_cache
    if _pipeline_cache is None:
        with _global_lock:
            if _pipeline_cache is None:
                _pipeline_cache = QueryCache()
    return _pipeline_cache


def clear_all_caches() -> int:
    """모든 캐시(Level 1 + Level 2)를 초기화한다.

    Returns:
        총 제거된 엔트리 수
    """
    total = 0
    total += get_search_cache().clear()
    total += get_pipeline_cache().clear()
    logger.info("전체 캐시 초기화 완료: %d건", total)
    return total

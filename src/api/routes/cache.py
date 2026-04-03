"""캐시 관리 API 라우트 — SPEC-RAG-004"""

from fastapi import APIRouter
from pydantic import BaseModel

from src.cache import clear_all_caches, get_search_cache, get_pipeline_cache

router = APIRouter()


class CacheStatsResponse(BaseModel):
    """캐시 통계 응답"""
    search_cache: dict
    pipeline_cache: dict


class CacheClearResponse(BaseModel):
    """캐시 초기화 응답"""
    message: str
    cleared_entries: int


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats() -> CacheStatsResponse:
    """캐시 통계 조회 (Level 1 + Level 2)"""
    return CacheStatsResponse(
        search_cache=get_search_cache().stats(),
        pipeline_cache=get_pipeline_cache().stats(),
    )


@router.delete("/cache", response_model=CacheClearResponse)
async def clear_cache() -> CacheClearResponse:
    """모든 캐시 초기화 (Level 1 + Level 2)"""
    cleared = clear_all_caches()
    return CacheClearResponse(
        message="캐시가 초기화되었습니다",
        cleared_entries=cleared,
    )

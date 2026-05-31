"""Shared FastAPI dependencies — Qdrant 클라이언트 싱글톤 프로바이더.

Qdrant 로컬 파일 모드(`QdrantClient(path=...)`)는 동일 폴더에 대한 동시
클라이언트 인스턴스를 허용하지 않는다. FastAPI 프로세스 안에서 검색·인덱싱·
에이전트가 각자 별도 클라이언트를 만들면 다음 에러가 발생한다:

    Storage folder ... is already accessed by another instance of Qdrant client.

이 모듈은 프로세스 전역 단일 QdrantClient를 제공해 모든 라우트가 공유하도록 한다.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

from qdrant_client import QdrantClient

_client: Optional[QdrantClient] = None
_lock = threading.Lock()


def get_qdrant_client() -> QdrantClient:
    """동일 FastAPI 프로세스 내에서 단일 QdrantClient를 공유한다.

    - `QDRANT_LOCATION` 환경변수가 `:memory:` 또는 `http*` URL이면 location 모드
    - 그 외(파일 경로)는 path 모드
    - 첫 호출 시 더블 체크 락(double-checked locking) 으로 스레드 안전 초기화
    """
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                location = os.getenv("QDRANT_LOCATION", "./qdrant_data")
                if location == ":memory:" or location.startswith("http"):
                    _client = QdrantClient(location=location)
                else:
                    _client = QdrantClient(path=location)
    return _client


def reset_qdrant_client() -> None:
    """싱글톤 초기화 (테스트 격리·셧다운 훅 용).

    기존 클라이언트가 있으면 close()를 시도한 후 None으로 재설정한다.
    close() 실패는 무시한다 (테스트 환경에서 이미 닫혀 있을 수 있음).
    """
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None

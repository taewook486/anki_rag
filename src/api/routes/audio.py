"""오디오 스트리밍 API 라우트 - SPEC-RAG-002 REQ-004"""

import hashlib
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

router = APIRouter()


def _resolve_audio_path(audio_id: str) -> Path:
    """audio_id(hex hash)로 실제 파일 경로 복원

    audio_id는 audio_path의 MD5 hex digest.
    data/media/ 하위 전체를 탐색하여 매칭 파일 반환.
    """
    media_root = Path(os.getenv("AUDIO_MEDIA_DIR", "./data/media"))
    if not media_root.exists():
        raise HTTPException(status_code=404, detail="미디어 디렉토리 없음")

    for ext in (".mp3", ".ogg", ".wav"):
        for path in media_root.rglob(f"*{ext}"):
            file_id = hashlib.md5(str(path).encode()).hexdigest()
            if file_id == audio_id:
                return path

    raise HTTPException(status_code=404, detail=f"오디오 파일 없음: {audio_id}")


@router.get("/audio/{audio_id}")
async def stream_audio(audio_id: str) -> FileResponse:
    """
    오디오 파일 스트리밍

    - **audio_id**: audio_path의 MD5 hex digest
    - HTTP Range 요청 지원 (partial download)
    - Content-Type: audio/mpeg (MP3) / audio/ogg / audio/wav
    """
    path = _resolve_audio_path(audio_id)

    content_type_map = {
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".wav": "audio/wav",
    }
    content_type = content_type_map.get(path.suffix.lower(), "audio/mpeg")

    etag = hashlib.md5(f"{path}{path.stat().st_mtime}".encode()).hexdigest()

    return FileResponse(
        path=str(path),
        media_type=content_type,
        headers={
            "ETag": f'"{etag}"',
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
        },
    )


@router.get("/audio/id/{audio_path:path}")
async def get_audio_id(audio_path: str) -> Response:
    """
    audio_path 문자열로 audio_id(MD5) 조회

    Streamlit에서 audio_path -> audio_id 변환에 사용.
    """
    audio_id = hashlib.md5(audio_path.encode()).hexdigest()
    return Response(content=audio_id, media_type="text/plain")

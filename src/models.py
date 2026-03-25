"""데이터 모델 - Pydantic 기반"""

from typing import Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Anki 노트 문서 모델"""

    # 필수 필드
    word: str = Field(..., description="단어 또는 표현")
    meaning: str = Field(..., description="뜻 또는 번역")
    source: str = Field(..., description="데이터 출처 (toefl, xfer, sentences 등)")
    deck: str = Field(..., description="덱 이름")

    # 선택적 필드
    pronunciation: Optional[str] = Field(None, description="발음 기호")
    example: Optional[str] = Field(None, description="예문")
    example_translation: Optional[str] = Field(None, description="예문 번역")
    tags: list[str] = Field(default_factory=list, description="태그 목록")
    note_type: Optional[str] = Field(None, description="노트 타입 (Simple Model, Basic 등)")
    audio_path: Optional[str] = Field(None, description="오디오 파일 경로")


class SearchResult(BaseModel):
    """검색 결과 모델"""

    document: Document = Field(..., description="검색된 문서")
    score: float = Field(..., description="관련성 점수")
    rank: int = Field(..., ge=1, description="검색 결과 순위")

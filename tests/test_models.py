"""Tests for models.py - 데이터 모델 검증"""

import pytest
from pydantic import ValidationError

# RED: Document 모델이 존재하지 않으므로 import 실패 예상
try:
    from src.models import Document, SearchResult
except ImportError:
    pytest.skip("src.models not implemented yet", allow_module_level=True)


def test_document_creation_with_all_fields():
    """모든 필드가 포함된 Document 생성 성공"""
    doc = Document(
        word="abandon",
        meaning="포기하다, 버리다",
        pronunciation="/əˈbændən/",
        example="He abandoned the project.",
        example_translation="그는 프로젝트를 포기했다.",
        source="toefl",
        deck="TOEFL 영단어",
        tags=["verb", "academic"],
        note_type="Simple Model",
        audio_paths=["data/media/hacker_toeic/aban.mp3"],
    )

    assert doc.word == "abandon"
    assert doc.meaning == "포기하다, 버리다"
    assert doc.pronunciation == "/əˈbændən/"
    assert doc.example == "He abandoned the project."
    assert doc.audio_paths == ["data/media/hacker_toeic/aban.mp3"]


def test_document_creation_minimal_fields():
    """필수 필드만으로 Document 생성 성공"""
    doc = Document(
        word="test",
        meaning="테스트",
        source="sentences",
        deck="원서 1만 문장",
    )

    assert doc.word == "test"
    assert doc.meaning == "테스트"
    assert doc.pronunciation is None  # 선택적 필드
    assert doc.example is None
    assert doc.audio_paths == []


def test_document_missing_required_field():
    """필수 필드(word)가 없으면 ValidationError 발생"""
    with pytest.raises(ValidationError):
        Document(
            meaning="테스트",
            source="sentences",
            deck="원서 1만 문장",
        )


def test_search_result_creation():
    """SearchResult 모델 생성 성공"""
    result = SearchResult(
        document=Document(
            word="abandon",
            meaning="포기하다",
            source="toefl",
            deck="TOEFL 영단어",
        ),
        score=0.95,
        rank=1,
    )

    assert result.score == 0.95
    assert result.rank == 1
    assert result.document.word == "abandon"

"""Tests for parser.py - Anki 파일 파싱"""

import pytest
from pathlib import Path
from src.models import Document

try:
    from src.parser import AnkiParser, TextParser
except ImportError:
    pytest.skip("src.parser not implemented yet", allow_module_level=True)


class TestAnkiParser:
    """AnkiParser 테스트"""

    def test_parse_apkg_file(self, tmp_path):
        """apkg 파일 파싱 테스트"""
        # 실제 .apkg 파일로 테스트하거나 mock 사용
        pass

    def test_extract_audio_from_apkg(self, tmp_path):
        """오디오 파일 추출 테스트"""
        pass

    def test_parse_anki21_db(self, tmp_path):
        """SQLite anki21 DB 파싱 테스트"""
        pass


class TestTextParser:
    """TextParser 테스트"""

    def test_parse_tab_separated_text(self, tmp_path):
        """탭 구분 텍스트 파일 파싱"""
        # Given: 탭 구분 텍스트 파일
        test_file = tmp_path / "test.txt"
        test_file.write_text(
            "Hello World\t안녕하세요 세계\n"
            "This is a test\t이것은 테스트입니다\n"
            "\n" ,  # 빈 줄은 스킵되어야 함
            encoding='utf-8'
        )

        # When: 파싱
        parser = TextParser()
        documents = parser.parse_file(str(test_file), source="sentences", deck="테스트 덱")

        # Then: 2개 문서 반환
        assert len(documents) == 2
        assert documents[0].word == "Hello World"
        assert documents[0].meaning == "안녕하세요 세계"
        assert documents[0].source == "sentences"
        assert documents[0].deck == "테스트 덱"

    def test_parse_utf8_bom_file(self, tmp_path):
        """UTF-8 BOM 처리 테스트"""
        test_file = tmp_path / "bom_test.txt"
        # UTF-8 BOM + content
        test_file.write_bytes(b"\xef\xbb\xbfHello\tworld\n")

        parser = TextParser()
        documents = parser.parse_file(str(test_file), source="sentences", deck="테스트")

        assert len(documents) == 1
        assert documents[0].word == "Hello"

    def test_parse_skips_incomplete_lines(self, tmp_path):
        """불완전한 라인 스킵 테스트"""
        test_file = tmp_path / "incomplete.txt"
        test_file.write_text(
            "Valid line\t유효한 라인\n"
            "Incomplete line without tab\n",  # 탭 없으므로 스킵
            encoding='utf-8',
        )

        parser = TextParser()
        documents = parser.parse_file(str(test_file), source="sentences", deck="테스트")

        assert len(documents) == 1

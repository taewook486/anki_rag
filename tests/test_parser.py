"""Tests for parser.py - Anki 파일 파싱"""

import io
import json
import sqlite3
import zipfile

import pytest
from pathlib import Path
from src.models import Document

try:
    from src.parser import AnkiParser, TextParser
except ImportError:
    pytest.skip("src.parser not implemented yet", allow_module_level=True)


# ── 테스트용 .apkg 생성 헬퍼 ──────────────────────────────────────────────────

def _make_apkg(
    tmp_path,
    db_name: str = "collection.anki21",
    notes: list[dict] | None = None,
    deck_name: str = "Test Deck",
    with_audio: bool = False,
) -> Path:
    """최소한의 유효한 .apkg 파일 생성"""
    models = {
        "1": {
            "id": 1,
            "name": "Basic",
            "flds": [
                {"name": "Front", "ord": 0},
                {"name": "뜻",    "ord": 1},
                {"name": "발음",  "ord": 2},
                {"name": "예문",  "ord": 3},
            ],
        }
    }
    decks = {"1": {"id": 1, "name": deck_name}}

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE col (id INTEGER, crt INTEGER, mod INTEGER, scm INTEGER, "
        "ver INTEGER, dty INTEGER, usn INTEGER, ls INTEGER, conf TEXT, "
        "models TEXT, decks TEXT, dconf TEXT, tags TEXT)"
    )
    conn.execute(
        "INSERT INTO col VALUES (1,0,0,0,11,0,0,0,'{}',?,?,'{}','{}')",
        (json.dumps(models), json.dumps(decks)),
    )
    conn.execute(
        "CREATE TABLE notes (id INTEGER, guid TEXT, mid INTEGER, mod INTEGER, "
        "usn INTEGER, tags TEXT, flds TEXT, sfld TEXT, csum INTEGER, "
        "flags INTEGER, data TEXT)"
    )
    for note in (notes or []):
        conn.execute(
            "INSERT INTO notes VALUES (?,''  ,1,0,0,?,?,''  ,0,0,'')",
            (note["id"], note.get("tags", ""), "\x1f".join(note["fields"])),
        )
    conn.commit()
    db_bytes = conn.serialize()
    conn.close()

    apkg_path = tmp_path / "test.apkg"
    with zipfile.ZipFile(apkg_path, "w") as zf:
        zf.writestr(db_name, db_bytes)
        media_data: dict[str, str] = {}
        if with_audio:
            media_data = {"0": "hello.mp3"}
            zf.writestr("0", b"fake_audio")
        zf.writestr("media", json.dumps(media_data))
    return apkg_path


# ── AnkiParser 테스트 ─────────────────────────────────────────────────────────

class TestAnkiParser:
    """AnkiParser 테스트"""

    def test_parse_apkg_file(self, tmp_path):
        """기본 apkg 파싱 — word/meaning 정상 추출"""
        apkg = _make_apkg(
            tmp_path,
            notes=[
                {"id": 1, "fields": ["abandon", "포기하다", "/əˈbændən/", "He abandoned the car."]},
                {"id": 2, "fields": ["acquire",  "얻다",      "",            ""]},
            ],
            deck_name="TOEFL 영단어",
        )
        parser = AnkiParser(extract_media=False)
        docs = parser.parse_file(str(apkg), source="toefl")

        assert len(docs) == 2
        assert docs[0].word == "abandon"
        assert docs[0].meaning == "포기하다"
        assert docs[0].pronunciation == "/əˈbændən/"
        assert docs[0].example == "He abandoned the car."
        assert docs[0].source == "toefl"
        assert docs[0].deck == "TOEFL 영단어"
        assert docs[1].word == "acquire"

    def test_parse_anki21_db(self, tmp_path):
        """collection.anki21 파일 우선 사용"""
        apkg = _make_apkg(
            tmp_path,
            db_name="collection.anki21",
            notes=[{"id": 1, "fields": ["give up", "포기하다", "", ""]}],
        )
        docs = AnkiParser(extract_media=False).parse_file(str(apkg), source="phrasal")
        assert len(docs) == 1
        assert docs[0].word == "give up"

    def test_parse_anki2_fallback(self, tmp_path):
        """collection.anki21 없을 때 collection.anki2 fallback"""
        apkg = _make_apkg(
            tmp_path,
            db_name="collection.anki2",
            notes=[{"id": 1, "fields": ["run out", "바닥나다", "", ""]}],
        )
        docs = AnkiParser(extract_media=False).parse_file(str(apkg), source="phrasal")
        assert len(docs) == 1
        assert docs[0].word == "run out"

    def test_extract_audio_from_apkg(self, tmp_path):
        """[sound:파일명] 패턴 오디오 추출"""
        apkg = _make_apkg(
            tmp_path,
            notes=[
                {"id": 1, "fields": ["hello", "안녕", "[sound:hello.mp3]", ""]},
            ],
            with_audio=True,
        )
        parser = AnkiParser(extract_media=True, media_output_dir=str(tmp_path / "media"))
        docs = parser.parse_file(str(apkg), source="hacker_toeic")

        assert len(docs) == 1
        assert docs[0].audio_path is not None
        assert "hello.mp3" in docs[0].audio_path

    def test_skip_notes_without_word_or_meaning(self, tmp_path):
        """word 또는 meaning이 없는 노트 스킵"""
        apkg = _make_apkg(
            tmp_path,
            notes=[
                {"id": 1, "fields": ["", "뜻이 있지만 단어 없음", "", ""]},
                {"id": 2, "fields": ["단어만 있음", "", "", ""]},
                {"id": 3, "fields": ["valid", "유효함", "", ""]},
            ],
        )
        docs = AnkiParser(extract_media=False).parse_file(str(apkg), source="toefl")
        assert len(docs) == 1
        assert docs[0].word == "valid"

    def test_strip_html_from_fields(self, tmp_path):
        """HTML 태그 제거"""
        apkg = _make_apkg(
            tmp_path,
            notes=[
                {"id": 1, "fields": ["<b>abandon</b>", "<span>포기하다</span>", "", ""]},
            ],
        )
        docs = AnkiParser(extract_media=False).parse_file(str(apkg), source="toefl")
        assert docs[0].word == "abandon"
        assert docs[0].meaning == "포기하다"

    def test_tags_parsed(self, tmp_path):
        """노트 태그 파싱"""
        apkg = _make_apkg(
            tmp_path,
            notes=[{"id": 1, "fields": ["word", "meaning", "", ""], "tags": "toefl vocab"}],
        )
        docs = AnkiParser(extract_media=False).parse_file(str(apkg), source="toefl")
        assert "toefl" in docs[0].tags
        assert "vocab" in docs[0].tags


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

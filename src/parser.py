"""파서 - Anki 파일 및 텍스트 파일 파싱"""

import zipfile
import sqlite3
import json
import re
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup

from src.models import Document


class AnkiParser:
    """Anki .apkg 패키지 파일 파서"""

    def __init__(self, extract_media: bool = True, media_output_dir: str = "data/media"):
        """
        Args:
            extract_media: 오디오 파일 추출 여부
            media_output_dir: 오디오 파일 출력 디렉토리
        """
        self.extract_media = extract_media
        self.media_output_dir = Path(media_output_dir)

    def parse_file(self, apkg_path: str, source: str) -> list[Document]:
        """
        .apkg 파일을 파싱하여 Document 리스트 반환

        Args:
            apkg_path: .apkg 파일 경로
            source: 데이터 출처 식별자 (toefl, xfer, hacker_toeic 등)

        Returns:
            Document 리스트
        """
        apkg_path = Path(apkg_path)

        with zipfile.ZipFile(apkg_path, "r") as zf:
            # SQLite DB 파일명 확인 (anki21 우선)
            db_filename = self._find_database(zf)
            if not db_filename:
                raise ValueError("No valid Anki database found in .apkg file")

            # 메타데이터 로드
            models_json, decks_json = self._load_metadata(zf, db_filename)

            # 오디오 추출
            media_mapping = {}
            if self.extract_media:
                media_mapping = self._extract_media_files(zf, source)

            # DB 파싱
            with zf.open(db_filename) as db_file:
                documents = self._parse_database(
                    db_file, models_json, decks_json, source, media_mapping
                )

        return documents

    def _find_database(self, zf: zipfile.ZipFile) -> Optional[str]:
        """ZIP 내에서 데이터베이스 파일 찾기"""
        # anki21 우선, 없으면 anki2
        for name in ["collection.anki21", "collection.anki2"]:
            if name in zf.namelist():
                return name
        return None

    def _load_metadata(self, zf: zipfile.ZipFile, db_filename: str) -> tuple[dict, dict]:
        """col 테이블에서 models와 decks JSON 로드"""
        with zf.open(db_filename) as db_file:
            db_bytes = db_file.read()
        conn = sqlite3.connect(":memory:")
        conn.deserialize(db_bytes)
        try:
            row = conn.execute("SELECT models, decks FROM col LIMIT 1").fetchone()
            if not row:
                return {}, {}
            return json.loads(row[0]), json.loads(row[1])
        except Exception:
            return {}, {}
        finally:
            conn.close()

    def _extract_media_files(self, zf: zipfile.ZipFile, source: str) -> dict[str, str]:
        """
        미디어 파일 추출

        Returns:
            {번호: 원본파일명} 매핑
        """
        media_json_path = "media"
        if media_json_path not in zf.namelist():
            return {}

        # media JSON 파싱: {"0": "file.mp3", "1": "sound.ogg", ...}
        with zf.open(media_json_path) as f:
            media_json = json.load(f)
            # Anki는 JSON을 문자열로 저장하므로 파싱 필요
            if isinstance(media_json, str):
                media_json = json.loads(media_json)

        # 미디어 파일 추출
        output_dir = self.media_output_dir / source
        output_dir.mkdir(parents=True, exist_ok=True)

        file_mapping = {}
        for num_str, filename in media_json.items():
            source_path = str(num_str)  # ZIP 내의 파일명은 숫자
            if source_path in zf.namelist():
                target_path = output_dir / filename
                with zf.open(source_path) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())
                file_mapping[source_path] = str(target_path)

        return file_mapping

    def _parse_database(
        self,
        db_file,
        models_json: dict,
        decks_json: dict,
        source: str,
        media_mapping: dict[str, str],
    ) -> list[Document]:
        """SQLite DB 파싱"""
        db_bytes = db_file.read()
        conn = sqlite3.connect(":memory:")
        conn.deserialize(db_bytes)
        documents = []
        try:
            deck_name = self._get_deck_name(decks_json, source)
            rows = conn.execute(
                "SELECT id, flds, mid, tags FROM notes"
            ).fetchall()
            for _note_id, flds_raw, mid, tags in rows:
                fields = flds_raw.split("\x1f")
                model = models_json.get(str(mid), {})
                field_names = [f["name"] for f in model.get("flds", [])]
                field_map = dict(zip(field_names, fields))

                word = self._pick_field(field_map, ["Front", "Question", "단어"])
                meaning = self._pick_field(field_map, ["뜻", "Back", "Answer"])
                if not word or not meaning:
                    continue

                pronunciation = self._pick_field(field_map, ["발음", "Pronunciation"]) or None
                example = self._pick_field(field_map, ["예문", "Example"]) or None
                example_translation = (
                    self._pick_field(field_map, ["예문 뜻", "Example Translation"]) or None
                )
                audio_path = self._extract_audio_from_field(flds_raw, media_mapping)
                tag_list = [t for t in (tags or "").split() if t]

                documents.append(
                    Document(
                        word=self._strip_html(word),
                        meaning=self._strip_html(meaning),
                        pronunciation=self._strip_html(pronunciation) if pronunciation else None,
                        example=self._strip_html(example) if example else None,
                        example_translation=(
                            self._strip_html(example_translation) if example_translation else None
                        ),
                        source=source,
                        deck=deck_name,
                        tags=tag_list,
                        audio_path=audio_path,
                    )
                )
        finally:
            conn.close()
        return documents

    def _get_deck_name(self, decks_json: dict, fallback: str) -> str:
        """decks JSON에서 첫 번째 비기본 덱 이름 반환"""
        for deck_info in decks_json.values():
            name = deck_info.get("name", "")
            if name and name != "Default":
                return name
        return fallback

    def _pick_field(self, field_map: dict, candidates: list[str]) -> str:
        """후보 필드명 중 첫 번째 존재하는 값 반환"""
        for name in candidates:
            val = field_map.get(name, "").strip()
            if val:
                return val
        return ""

    def _strip_html(self, text: str) -> str:
        """HTML 태그 제거"""
        return BeautifulSoup(text, "html.parser").get_text().strip()

    def _extract_audio_from_field(self, flds: str, media_mapping: dict) -> Optional[str]:
        """필드에서 [sound:filename] 패턴 추출"""
        pattern = r"\[sound:([^\]]+)\]"
        match = re.search(pattern, flds)
        if match:
            filename = match.group(1)
            # media_mapping에서 해당 파일 경로 찾기
            for num, path in media_mapping.items():
                if filename in path:
                    return path
        return None


class TextParser:
    """탭 구분 텍스트 파일 파서 (10000.txt 등)"""

    def __init__(self):
        pass

    def parse_file(
        self, file_path: str, source: str, deck: str
    ) -> list[Document]:
        """
        텍스트 파일 파싱

        Args:
            file_path: 텍스트 파일 경로
            source: 데이터 출처
            deck: 덱 이름

        Returns:
            Document 리스트
        """
        file_path = Path(file_path)
        documents = []

        # UTF-8 BOM 처리를 위해 utf-8-sig 인코딩 사용
        with open(file_path, "r", encoding="utf-8-sig") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 빈 줄 스킵
                    continue

                # 탭으로 분리
                parts = line.split("\t")
                if len(parts) < 2:  # 최소 2개 필드 필요
                    continue

                word = parts[0].strip()
                meaning = parts[1].strip()

                if not word or not meaning:  # 빈 필드 스킵
                    continue

                doc = Document(
                    word=word,
                    meaning=meaning,
                    source=source,
                    deck=deck,
                )
                documents.append(doc)

        return documents

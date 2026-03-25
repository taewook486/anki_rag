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
        # 실제 구현에서는 SQLite DB에서 직접 읽어야 함
        # 여기서는 stub으로 반환
        return {}, {}

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
        # 메모리에서 DB 열기
        with sqlite3.connect(":memory:") as conn:
            # DB 파일 내용을 메모리 DB로 로드
            db_content = db_file.read()
            conn.executescript(f"CREATE TABLE notes AS SELECT * FROM (VALUES ('{db_content}'))")

            # 실제 구현에서는 notes 테이블 파싱
            # 여기서는 stub 반환
            return []

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

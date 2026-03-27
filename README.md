# Anki RAG

영어 학습 특화 RAG (Retrieval-Augmented Generation) 시스템

## 개요

Anki 플래시카드 데이터(.apkg)를 파싱하여 벡터 DB(Qdrant)에 인덱싱하고, BGE-M3 하이브리드 검색 + Claude API를 통해 영어 단어 검색 및 질의응답을 제공합니다.

## 기능

- **Anki 파싱**: .apkg 파일 파싱 (anki21/anki2 SQLite DB)
- **오디오 추출**: .apkg 내 오디오 파일 자동 추출
- **하이브리드 검색**: BGE-M3 Dense + Sparse 벡터 검색
- **RAG 질의**: Claude API (claude-sonnet-4-6) 기반 응답 생성
- **오디오 재생**: 단어 발음 재생 지원

## 설치

```bash
# 의존성 설치
pip install -e .

# 또는 dev 모드 (테스트 포함)
pip install -e ".[dev]"
```

## 사용법

### 1. 인덱싱

```bash
python -m anki_rag index --data-dir ./data
```

### 2. 검색

```bash
# 단순 검색
python -m anki_rag search "abandon" --source toefl --top-k 5

# 검색 + 오디오 재생
python -m anki_rag search "abandon" --play-audio
```

### 3. RAG 질의

```bash
python -m anki_rag query "abandon의 뜻과 예문을 알려줘"
```

### 4. 대화형 모드

```bash
python -m anki_rag chat
```

## Qdrant 실행

### Docker (추천)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### In-memory (테스트용)

코드에서 `location=":memory:"` 사용

## 환경변수

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

## 테스트

```bash
pytest
```

## 프로젝트 구조

```
data/
├── *.apkg              # Anki 패키지 파일
└── 10000.txt           # 원서 문장

src/
├── models.py           # 데이터 모델 (Pydantic)
├── parser.py           # Anki 파서
├── embedder.py         # BGE-M3 임베딩
├── indexer.py          # Qdrant 인덱서
├── retriever.py        # 하이브리드 검색
├── audio.py            # 오디오 플레이어
├── rag.py              # RAG 파이프라인
└── __main__.py         # CLI

tests/
└── test_*.py           # 테스트 파일
```

## 문서

설계서는 `doc/설계서.md`에 있습니다.

PDF 생성 (pandoc + wkhtmltopdf 필요):

```bash
pandoc doc/설계서.md -o doc/설계서.pdf --pdf-engine=wkhtmltopdf -V margin-top=25 -V margin-bottom=25 -V margin-left=25 -V margin-right=25 --metadata title="설계서"
```

## 기술 스택

- Python 3.12+
- BGE-M3 (FlagEmbedding)
- Qdrant
- Claude API (anthropic)
- Pydantic
- Click

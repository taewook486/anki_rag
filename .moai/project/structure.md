# Project Structure

## Directory Layout

```
anki_rag/
├── src/                          # 메인 소스 코드
│   ├── __init__.py
│   ├── __main__.py              # CLI 진입점 (Click)
│   ├── models.py                # Pydantic 데이터 모델
│   ├── parser.py                # Anki/Text 파일 파서
│   ├── embedder.py              # BGE-M3 임베딩
│   ├── indexer.py               # Qdrant 인덱싱
│   ├── retriever.py             # 하이브리드 검색
│   ├── audio.py                 # 오디오 재생
│   ├── rag.py                   # Claude API RAG 파이프라인
│   ├── api/                     # FastAPI 백엔드
│   │   ├── main.py              # FastAPI 앱 초기화
│   │   └── routes/
│   │       ├── search.py        # POST /api/search
│   │       └── query.py         # POST /api/query
│   └── web/                     # Streamlit 프론트엔드
│       ├── app.py               # 메인 Streamlit 앱
│       └── pages/               # 페이지 (stub)
├── tests/                       # 테스트
│   ├── test_models.py
│   ├── test_parser.py
│   ├── test_embedder.py
│   ├── test_indexer.py
│   ├── test_retriever.py
│   ├── test_audio.py
│   └── test_rag.py
├── data/                        # 데이터 디렉토리
│   ├── *.apkg                   # Anki 패키지 (5개)
│   ├── 10000.txt                # 원서 1만 문장
│   └── media/                   # 추출된 오디오 파일
├── pyproject.toml               # 프로젝트 설정
├── CLAUDE.md                    # MoAI 지시문
└── .moai/                       # MoAI 설정
    ├── config/                  # 설정 파일
    ├── specs/                   # SPEC 문서
    └── project/                 # 프로젝트 문서
```

## Module Dependencies

```
models.py (base - 의존성 없음)
    ↓
parser.py → beautifulsoup4, zipfile, sqlite3
embedder.py → torch, FlagEmbedding
indexer.py → qdrant_client
retriever.py → embedder.py, qdrant_client
audio.py → platform, subprocess
rag.py → retriever.py, anthropic
__main__.py → click, 모든 모듈
api/ → fastapi, retriever.py, rag.py
web/ → streamlit, requests (API 호출)
```

## Data Flow

```
[Indexing]
.apkg/.txt → parser → List[Document] → embedder → List[EmbeddingResult] → indexer → Qdrant

[Search]
query → embedder → retriever (Qdrant search) → List[SearchResult]

[RAG]
query → retriever → context → rag (Claude API) → answer

[Web]
Streamlit → HTTP → FastAPI → retriever/rag → response
```

## Entry Points

| 인터페이스 | 진입점 | 명령어 |
|------------|--------|--------|
| CLI | src/__main__.py | `python -m src index/search/query/chat` |
| API | src/api/main.py | `uvicorn src.api.main:app --port 8000` |
| Web | src/web/app.py | `streamlit run src/web/app.py` |

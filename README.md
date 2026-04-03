# Anki RAG

영어 학습 특화 RAG (Retrieval-Augmented Generation) 시스템

## 개요

Anki 플래시카드 데이터(.apkg)를 파싱하여 벡터 DB(Qdrant)에 인덱싱하고, BGE-M3 RRF 하이브리드 검색 + Multi-LLM을 통해 영어 단어 검색 및 질의응답을 제공합니다. OpenAI 호환 API(GLM, OpenRouter 등)와 Anthropic Claude를 선택적으로 지원합니다.

## 기능

- **Anki 파싱**: .apkg 파일 파싱 (anki21/anki2 SQLite DB)
- **오디오 추출**: .apkg 내 오디오 파일 자동 추출
- **RRF 하이브리드 검색**: Dense + Sparse 검색 + RRF Fusion (k=60) + Exact Match 부스팅 + 단어 중복 제거
- **Multi-LLM 지원**: OpenAI 호환 API (GLM, OpenRouter 등) + Anthropic Claude 선택적 지원
- **Web GUI**: FastAPI REST API + Streamlit 프론트엔드
- **Few-shot 프롬프트**: 일관된 답변 포맷 (단어/발음/뜻/예문/출처)
- **오디오 재생**: 단어 발음 재생 지원

## 설치

```bash
# 의존성 설치
pip install -e .

# 또는 dev 모드 (테스트 포함)
pip install -e ".[dev]"
```

## 사용법

### Web GUI

```bash
# 방법 1: start.bat으로 동시 실행 (Windows)
start.bat

# 방법 2: 개별 실행

# FastAPI 서버
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

# Streamlit 프론트엔드
streamlit run src/web/app.py
```

### CLI

#### 1. 인덱싱

```bash
python -m anki_rag index --data-dir ./data
```

#### 2. 검색

```bash
# 단순 검색
python -m anki_rag search "abandon" --source toefl --top-k 5

# 검색 + 오디오 재생
python -m anki_rag search "abandon" --play-audio
```

#### 3. RAG 질의

```bash
python -m anki_rag query "abandon의 뜻과 예문을 알려줘"
```

#### 4. 대화형 모드

```bash
python -m anki_rag chat
```

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search` | POST | 하이브리드 검색 (exclude_sentences, deduplicate 지원) |
| `/api/query` | POST | RAG 질의응답 |
| `/api/audio/{id}` | GET | 오디오 스트리밍 |
| `/api/index` | POST | 백그라운드 인덱싱 |
| `/api/index/status` | GET | 인덱싱 상태 조회 |

API 문서는 서버 실행 후 `http://127.0.0.1:8000/docs`에서 확인할 수 있습니다.

## 환경변수

```bash
# OpenAI 호환 API (GLM, OpenRouter 등)
LLM_BASE_URL=https://your-endpoint
LLM_API_KEY=your-api-key
LLM_MODEL=glm-5

# Anthropic API (선택, 우선순위 높음)
ANTHROPIC_API_KEY=sk-ant-...

# Qdrant
QDRANT_LOCATION=./qdrant_data
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
├── retriever.py        # 하이브리드 검색 (RRF + Exact Match + 중복 제거)
├── audio.py            # 오디오 플레이어
├── rag.py              # Multi-LLM RAG 파이프라인
├── __main__.py         # CLI
├── api/                # FastAPI REST API
│   ├── main.py
│   └── routes/
│       ├── search.py   # POST /api/search
│       ├── query.py    # POST /api/query
│       ├── audio.py    # GET /api/audio/{id}
│       └── index.py    # POST /api/index
└── web/                # Streamlit 프론트엔드
    └── app.py

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
- Multi-LLM (OpenAI 호환 API + Anthropic 선택)
- FastAPI
- Streamlit
- Pydantic
- Click

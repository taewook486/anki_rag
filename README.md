# Anki RAG

영어 학습 특화 RAG (Retrieval-Augmented Generation) 시스템

## 개요

Anki 플래시카드 데이터(.apkg)를 파싱하여 벡터 DB(Qdrant)에 인덱싱하고, BGE-M3 RRF 하이브리드 검색 + Multi-LLM을 통해 영어 단어 검색 및 질의응답을 제공합니다. Agentic AI Agent 구조(ReAct 패턴, Self-Correction)와 지식 그래프 기반 GraphRAG를 지원합니다.

## 기능

### v1.x (하이브리드 RAG)

- **Anki 파싱**: .apkg 파일 파싱 (anki21/anki2 SQLite DB)
- **오디오 추출**: .apkg 내 오디오 파일 자동 추출
- **RRF 하이브리드 검색**: Dense + Sparse 검색 + RRF Fusion (k=60) + Exact Match 부스팅 + 단어 중복 제거
- **Multi-LLM 지원**: OpenAI 호환 API (GLM, OpenRouter 등) + Anthropic Claude 선택적 지원
- **Agentic AI Agent**: ReAct 패턴 기반 멀티스텝 추론 및 Tool-use, Self-Correction
- **Few-shot 프롬프트**: 일관된 답변 포맷 (단어/발음/뜻/예문/출처) + hallucination 방지
- **응답 캐싱**: 2단계 LRU 캐시 (검색 캐시 + 파이프라인 캐시)
- **오디오 재생**: 단어 발음 재생 지원

### v2.0 (GraphRAG)

- **지식 그래프**: WordNet 기반 단어 유의어·반의어·파생어 관계 그래프 + 동시성 관계 엣지
- **그래프 영속화**: pickle 및 GraphML 이중 저장 (data/graph.pkl, data/graph.graphml)
- **GraphRAG Fusion**: Complex 쿼리 경로에서 하이브리드 검색과 그래프 기반 컨텍스트 자동 통합
- **그래프 API**: `/api/graph/related/{word}` (관계 타입 필터) + `/api/graph/stats` (통계)
- **Streamlit 지식 그래프 탭**: 인터랙티브 그래프 시각화 및 관계 탐색
- **Web UI**: FastAPI + Streamlit 기반 검색·채팅·그래프 탐색·관리 인터페이스

## 설치

### 1. 패키지 설치

```bash
# 의존성 설치
pip install -e .

# 또는 dev 모드 (테스트 포함)
pip install -e ".[dev]"
```

### 2. WordNet 코퍼스 다운로드 (GraphRAG 사용 시)

```bash
# Python 대화형 모드
python
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download('omw-1.4')  # 다국어 지원 (선택)
>>> exit()

# 또는 직접 명령어
python -m nltk.downloader wordnet omw-1.4
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
uv run python -m src index --data-dir ./data
```

#### 2. 검색

```bash
# 단순 검색
uv run python -m src search "abandon" --source toefl --top-k 5

# 덱 필터 + 오디오 재생
uv run python -m src search "abandon" --deck "TOEFL 영단어" --play-audio
```

#### 3. RAG 질의

```bash
# 일반 질의
uv run python -m src query "abandon의 뜻과 예문을 알려줘"

# 스트리밍 출력
uv run python -m src query "give up 관련 구동사를 알려줘" --stream
```

#### 4. 대화형 모드

```bash
uv run python -m src chat --stream
```

### 5. Web UI (FastAPI + Streamlit)

```bash
# Windows: 두 서버 동시 실행
start.bat

# 개별 실행
uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
uv run streamlit run src/web/app.py
```

**Streamlit 탭:**
- **검색**: 단어 검색 + 덱 필터 + 오디오 재생
- **채팅**: RAG 기반 대화
- **지식 그래프** (v2.0): 인터랙티브 단어 관계 그래프 시각화

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search` | POST | 하이브리드 검색 (exclude_sentences, deduplicate 지원) |
| `/api/query` | POST | RAG 질의응답 |
| `/api/audio/{id}` | GET | 오디오 스트리밍 |
| `/api/index` | POST | 백그라운드 인덱싱 |
| `/api/index/status` | GET | 인덱싱 상태 조회 |
| `/api/adaptive` | POST | 적응형 RAG (Simple/Moderate/Complex 전략 자동 선택, use_graph 파라미터 지원) |
| `/api/graph/related/{word}` | GET | 그래프 기반 단어 관계 조회 (관계 타입 필터: SYNONYM, ANTONYM, DERIVATION, CO_OCCURS) |
| `/api/graph/stats` | GET | 지식 그래프 통계 (노드/엣지 수, 관계 분포) |

API 문서는 서버 실행 후 `http://127.0.0.1:8000/docs`에서 확인할 수 있습니다.

## 환경변수

```bash
# LLM API (둘 중 하나 필수, Anthropic 우선)
ANTHROPIC_API_KEY=sk-ant-...         # Claude API
LLM_API_KEY=your-api-key             # OpenAI 호환 API (GLM, OpenRouter 등)
LLM_BASE_URL=https://your-endpoint
LLM_MODEL=glm-5

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
├── 10000.txt           # 원서 문장
└── media/              # 추출된 오디오 파일

src/
├── models.py           # 데이터 모델 (Pydantic)
├── parser.py           # Anki 파서
├── embedder.py         # BGE-M3 임베딩
├── indexer.py          # Qdrant 인덱서
├── retriever.py        # 하이브리드 검색 (RRF + Exact Match + 중복 제거)
├── audio.py            # 오디오 플레이어
├── rag.py              # Multi-LLM RAG 파이프라인
├── agent.py            # ReAct LearningAgent (Tool-use, Self-Correction)
├── adaptive.py         # 쿼리 분류 및 적응형 RAG 전략
├── graph.py            # 지식 그래프 (NetworkX MultiDiGraph, WordNet, GraphRAG Fusion)
├── __main__.py         # CLI 진입점
├── api/
│   ├── main.py         # FastAPI 앱
│   └── routes/         # search / query / index / audio / cache / agent / adaptive / graph
└── web/
    └── app.py          # Streamlit UI (검색, 채팅, 지식 그래프 탭)

doc/
├── 설계서.md           # 시스템 설계서
├── 설계서.hwp          # HWP 버전
└── md_to_hwp.py        # MD→HWP 변환 스크립트

tests/
└── test_*.py           # 테스트 파일
```

## 문서

설계서는 `doc/설계서.md`에 있습니다 (HWP / PDF 버전 포함).

문서 재생성:

```bash
# PDF (wkhtmltopdf 필요)
pandoc doc/설계서.md -o doc/설계서.pdf --pdf-engine=wkhtmltopdf \
  -V margin-top=25 -V margin-bottom=25 -V margin-left=25 -V margin-right=25

# HWP (한글 소프트웨어 + python-docx 필요)
python doc/md_to_hwp.py
```

## 기술 스택

| 범주 | 기술 |
|------|------|
| 언어 | Python 3.12+ |
| 임베딩 | BGE-M3 (FlagEmbedding) |
| 벡터 DB | Qdrant |
| LLM | Claude API / OpenAI 호환 (GLM 등) |
| API 서버 | FastAPI + uvicorn |
| Web UI | Streamlit |
| 그래프 | NetworkX (MultiDiGraph) |
| 데이터 모델 | Pydantic v2 |
| CLI | Click |
| 테스트 | pytest |

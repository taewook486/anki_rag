# SPEC-RAG-002: Web GUI 추가

**Status**: Completed
**Created**: 2026-03-20
**Author**: [I
**Parent SPEC**: SPEC-RAG-001
**Tech Stack**: FastAPI, Streamlit, Python 3.11+

---

## 1. 개요 (Overview)

기존 CLI 기반 Anki RAG 시스템(SPEC-RAG-001)에 웹 기반 GUI를 추가하여 브라우저에서 직관적인 인터페이스로 검색 및 질의 기능을 제공한다.

### 1.1 목표

- **검색 UI**: 단어 검색, 필터링, 결과 표시
- **RAG 채팅**: 대화형 질의응답 인터페이스
- **오디오 재생**: 웹에서 직접 발음 재생
- **인덱싱 상태**: 데이터 인덱싱 진행률 표시

### 1.2 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                       Browser                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Streamlit Frontend (UI)                    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │ HTTP
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  ┌──────────────────────────────────────────────────────┐  │
│  │         FastAPI Backend (REST API)                  │  │
│  │  - POST /api/search       (검색)                    │  │
│  │  - POST /api/query        (RAG 질의)                │  │
│  │  - GET  /api/audio/{id}   (오디오 스트리밍)        │  │
│  │  - POST /api/index        (인덱싱)                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                 │
│  ┌─────────────────────────┼───────────────────────────┐  │
│  │  Existing Modules       │                           │  │
│  │  (from SPEC-RAG-001)    │                           │  │
│  │  ┌──────────────────┐   │                           │  │
│  │  │ Retriever        │◄──┘                           │  │
│  │  │ RAG Pipeline     │◄──┘                           │  │
│  │  │ Audio Player     │◄──┘                           │  │
│  │  │ Qdrant Indexer   │◄──┘                           │  │
│  │  └──────────────────┘                               │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 기능 요구사항 (EARS Format)

### REQ-001: FastAPI 백엔드 서버

**WHEN** 사용자가 서버를 시작하면
**THE SYSTEM SHALL** FastAPI 기반 REST API 서버를 실행한다.

**수락 기준**:
- 기본 호스트: `127.0.0.1`
- 기본 포트: `8000`
- CORS 활성화 (Streamlit 통신)
- API 문서 자동 생성 (`/docs` 엔드포인트)

### REQ-002: 검색 API

**WHEN** 클라이언트가 POST `/api/search`를 요청하면
**THE SYSTEM SHALL** 하이브리드 검색을 수행하고 결과를 반환한다.

**수락 기준**:
```json
// Request
{
  "query": "abandon",
  "top_k": 10,
  "source_filter": "toefl"
}

// Response
{
  "results": [
    {
      "word": "abandon",
      "meaning": "포기하다, 버리다",
      "pronunciation": "/əˈbændən/",
      "example": "He abandoned the project.",
      "example_translation": "그는 프로젝트를 포기했다.",
      "source": "toefl",
      "deck": "TOEFL 영단어",
      "score": 0.95,
      "rank": 1,
      "audio_available": true
    }
  ]
}
```

### REQ-003: RAG 질의 API

**WHEN** 클라이언트가 POST `/api/query`를 요청하면
**THE SYSTEM SHALL** Claude API RAG 파이프라인을 실행하고 답변을 반환한다.

**수락 기준**:
```json
// Request
{
  "question": "abandon의 뜻과 예문을 알려줘",
  "top_k": 5,
  "source_filter": null
}

// Response
{
  "answer": "abandon은 '포기하다, 버리다'라는 뜻입니다...",
  "sources": [
    {"word": "abandon", "source": "toefl", "deck": "TOEFL 영단어"}
  ]
}
```

### REQ-004: 오디오 스트리밍 API

**WHEN** 클라이언트가 GET `/api/audio/{id}`를 요청하면
**THE SYSTEM SHALL** 오디오 파일을 스트리밍한다.

**수락 기준**:
- HTTP Range 요청 지원 (partial download)
- Content-Type: `audio/mpeg` (MP3)
- ETag 헤더로 캐싱 지원

### REQ-005: Streamlit 프론트엔드

**WHEN** 사용자가 웹 UI에 접속하면
**THE SYSTEM SHALL** Streamlit 기반 검색 및 질의 인터페이스를 제공한다.

**수락 기준**:
- **검색 페이지**: 검색창, 필터(checkbox), 결과 테이블
- **채팅 페이지**: 채팅 UI, 히스토리 표시
- **인덱싱 페이지**: 진행률 바, 상태 메시지
- **반응형 디자인**: 모바일 지원

---

## 3. 화면 설계 (Wireframe)

### 3.1 검색 페이지

```
┌─────────────────────────────────────────────────────────────┐
│  Anki RAG 검색                            [검색] [채팅] [관리]  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  검색어: [___________________] [🔍 검색]                    │
│                                                               │
│  필터:                                                         │
│  ☐ TOEFL  ☐ 편입  ☐ 해커스 토익  ☐ 구동사  ☐ 원서 문장     │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 결과 10건                                              │ │
│  ├─────────────────────────────────────────────────────────┤ │
│  │ abandon  |  포기하다, 버리다  │  🎵  │  toefl  │ 0.95 │ │
│  │ He abandoned the project.                              │ │
│  │                                                         │ │
│  │ give up  |  포기하다  │  🎵  │  phrasal  │ 0.87 │ │
│  │ ...                                                     │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 채팅 페이지

```
┌─────────────────────────────────────────────────────────────┐
│  Anki RAG 채팅                            [검색] [채팅] [관리]  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 채팅 기록                                              │ │
│  │ ──────────────────────────────────────────────────────  │ │
│  │ 👤 abandon의 뜻과 예문을 알려줘                        │ │
│  │                                                         │ │
│  │ 🤖 abandon은 '포기하다, 버리다'라는 뜻입니다...         │ │
│  │    [🎵 발음 듣기]                                      │ │
│  │                                                         │ │
│  │ 👤 give up과 비슷한 단어는?                             │ │
│  │                                                         │ │
│  │ 🤖 give up과 유사한 표현으로는...                       │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  [_________________________________] [Send] 🎵               │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 기술 의존성

```toml
# pyproject.toml 추가
[project.optional-dependencies]
web = [
    "fastapi>=0.104",
    "uvicorn[standard]>=0.24",
    "streamlit>=1.28",
    "python-multipart>=0.0.6",  # 파일 업로드
]
```

---

## 5. 파일 구조

```
src/
├── api/                    # FastAPI 백엔드
│   ├── __init__.py
│   ├── main.py            # FastAPI app
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── search.py      # /api/search
│   │   ├── query.py       # /api/query
│   │   └── audio.py       # /api/audio
│   └── static/            # 정적 파일 (오디오)
├── web/                   # Streamlit 프론트엔드
│   ├── __init__.py
│   ├── app.py            # Streamlit app
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── search.py     # 검색 페이지
│   │   ├── chat.py       # 채팅 페이지
│   │   └── admin.py      # 관리 페이지
│   └── components/
│       ├── __init__.py
│       └── result_card.py
└── __main__.py           # CLI (기존)
```

---

## 6. 구현 순서

| 단계 | 모듈 | 우선순위 |
|------|------|---------|
| 1 | `src/api/main.py` | P0 - FastAPI 서버 |
| 2 | `src/api/routes/search.py` | P0 - 검색 API |
| 3 | `src/api/routes/query.py` | P0 - RAG API |
| 4 | `src/api/routes/audio.py` | P1 - 오디오 스트리밍 |
| 5 | `src/web/app.py` | P1 - Streamlit 메인 |
| 6 | `src/web/pages/search.py` | P1 - 검색 페이지 |
| 7 | `src/web/pages/chat.py` | P2 - 채팅 페이지 |
| 8 | `src/web/pages/admin.py` | P2 - 관리 페이지 |

---

## 7. 수락 기준 (Given-When-Then)

### AC-001: FastAPI 서버 시작

```gherkin
Given 서버가 시작되지 않은 상태
When py -m src.api.main을 실행하면
Then "Uvicorn running on http://127.0.0.1:8000" 메시지가 출력된다
And http://localhost:8000/docs에 접속하면 API 문서가 표시된다
```

### AC-002: 검색 API

```gherkin
Given Qdrant에 데이터가 인덱싱되어 있다
When POST /api/search에 {"query": "abandon", "top_k": 5}를 요청하면
Then 200 OK 응답과 함께 results 배열이 반환된다
And results[0].word가 "abandon"을 포함한다
And results[0].score가 0.0~1.0 사이 값이다
```

### AC-003: Streamlit 검색 UI

```gherkin
Given Streamlit 서버가 실행 중
When 브라우저에서 http://localhost:8501에 접속하면
Then 검색창과 필터 체크박스가 표시된다
When "abandon"을 검색하면
Then 검색 결과 테이블에 10건 이하의 결과가 표시된다
```

---

## 8. 실행 방법

### 백엔드 서버 시작

```bash
py -m src.api.main
# http://localhost:8000
```

### 프론트엔드 시작

```bash
py -m streamlit run src/web/app.py
# http://localhost:8501
```

### 통합 시작 (추후)

```bash
py -m src.web.launch  # 두 서버 동시 시작
```

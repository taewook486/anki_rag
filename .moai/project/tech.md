# Technology Stack

## Core

| 기술 | 버전 | 용도 |
|------|------|------|
| Python | >= 3.11 | 런타임 |
| Pydantic | >= 2.7 | 데이터 모델 검증 |
| Click | >= 8.1 | CLI 프레임워크 |

## AI / ML

| 기술 | 버전 | 용도 |
|------|------|------|
| BGE-M3 (FlagEmbedding) | >= 1.2 | Dense + Sparse 임베딩 |
| PyTorch | >= 2.3 | 텐서 연산 |
| Anthropic SDK | >= 0.28 | Claude API 호출 |

## Data / Search

| 기술 | 버전 | 용도 |
|------|------|------|
| Qdrant | >= 1.9 | 벡터 데이터베이스 |
| BeautifulSoup4 | >= 4.12 | HTML 파싱 (Anki 필드) |
| SQLite3 | built-in | Anki DB 읽기 |

## Web

| 기술 | 버전 | 용도 |
|------|------|------|
| FastAPI | >= 0.104 | REST API 서버 |
| Uvicorn | >= 0.24 | ASGI 서버 |
| Streamlit | >= 1.28 | 웹 프론트엔드 |

## Dev Tools

| 기술 | 버전 | 용도 |
|------|------|------|
| pytest | >= 8.0 | 테스트 프레임워크 |
| pytest-cov | >= 4.1 | 커버리지 리포트 |
| ruff | >= 0.3 | 린팅 |
| mypy | >= 1.9 | 타입 체크 |

## Key Design Decisions

### 임베딩: BGE-M3
- Dense(1024차원) + Sparse(SPLADE) 동시 생성
- 다국어 지원 (영어 + 한국어 혼합 데이터에 적합)

### 벡터 DB: Qdrant
- Hybrid Search 네이티브 지원
- 필터링 (source, deck) 내장
- 현재 :memory: 모드, Docker 전환 가능

### LLM: Claude API (claude-sonnet-4-6)
- RAG 응답 생성 전용
- 스트리밍 지원
- ANTHROPIC_API_KEY 환경변수 필요

### 프론트엔드: Streamlit
- 빠른 프로토타이핑
- FastAPI 백엔드와 분리된 구조

### 오디오: 네이티브 명령어
- playsound 라이브러리 제거 (Python 3.12+ 호환성 문제)
- Windows: PowerShell WMPlayer, macOS: afplay, Linux: aplay/paplay

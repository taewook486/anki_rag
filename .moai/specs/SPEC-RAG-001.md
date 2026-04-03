# SPEC-RAG-001: Anki RAG 시스템

**Status**: Completed
**Created**: 2026-03-19
**Author**: [I
**Tech Stack**: Python 3.12+, BGE-M3, Qdrant, Claude API (claude-sonnet-4-6)

---

## 1. 개요 (Overview)

Anki 플래시카드 데이터(.apkg)와 텍스트 파일을 파싱하여 벡터 DB(Qdrant)에 인덱싱하고,
BGE-M3 Dense+Sparse 하이브리드 검색 + Claude API를 통해 영어 학습 특화 RAG 시스템을 구축한다.

### 1.1 데이터 인벤토리

| 파일명 | 포맷 | 카드 수 | source | 덱 명 |
|--------|------|---------|--------|-------|
| toefl_voca_v1.apkg | .anki2 | 5,308 | toefl | TOEFL 영단어 |
| xfer_voca_2022.apkg | .anki2 | 3,262 | xfer | 편입 영단어 2022 |
| _-forvo_-youglish_link.apkg | .anki21 | 1,227 | hacker_toeic | 해커스 토익 |
| --forvo-youglish_link.apkg | .anki21 | 2,438 | hacker_green | 해커스-초록이 |
| _-forvo-youglish_.apkg | .anki21 | 2,237 | phrasal | 구동사 |
| 10000.txt | tab-separated | 10,000 | sentences | 원서 1만 문장 |

총 **24,472개** Document

### 1.2 목표

- 영단어 검색: "abandon의 뜻은?" → 정의 + 예문 + 발음 정보 반환
- 예문 검색: "포기하다 영어 예문" → 관련 예문 + 출처 반환
- 유사 단어 검색: "give up 관련 구동사" → 의미적으로 유사한 구동사 목록
- 복합 질의: "TOEFL에 나오는 학술적인 단어 중 경제 관련" → 필터 + 시맨틱 검색

---

## 2. 기능 요구사항 (EARS Format)

### REQ-001: Anki 패키지 파싱

**WHEN** 사용자가 `.apkg` 파일 경로를 제공하면
**THE SYSTEM SHALL** 해당 파일을 ZIP으로 열고 내부의 `collection.anki21` 또는 `collection.anki2` SQLite DB를 파싱하여 노트 목록을 반환한다.

**수락 기준**:
- `.anki21` 파일 우선 사용, 없으면 `.anki2` fallback
- `notes` 테이블의 `flds` 필드를 `\x1f` (chr 31)로 분리하여 각 필드값 추출
- `col` 테이블의 `models` JSON에서 노트 타입별 필드명 매핑
- `col` 테이블의 `decks` JSON에서 덱 이름 추출
- HTML 태그 제거 후 순수 텍스트 저장 (선택적으로 HTML 보존 옵션)
- 파싱 실패 시 상세 오류 메시지 반환

### REQ-002: 텍스트 파일 파싱

**WHEN** 사용자가 탭 구분 텍스트 파일(10000.txt)을 제공하면
**THE SYSTEM SHALL** UTF-8 BOM (`utf-8-sig`) 인코딩으로 읽고 탭으로 분리하여 문장-번역 쌍을 반환한다.

**수락 기준**:
- UTF-8 BOM 처리 (`utf-8-sig` 인코딩)
- 탭(`\t`) 구분으로 필드 분리
- 최소 2개 필드(영문 문장, 한국어 번역) 보장
- 빈 줄 및 불완전한 행 스킵

### REQ-003: 노트 타입별 필드 매핑

**WHEN** 파서가 노트를 처리할 때
**THE SYSTEM SHALL** 노트 타입(Note Type)에 따라 아래 매핑을 적용하여 구조화된 Document를 생성한다.

| Note Type 패턴 | word 필드 | meaning 필드 | pronunciation | example |
|----------------|-----------|--------------|---------------|---------|
| Simple Model (toefl/xfer) | Question | Answer (HTML 포함) | Answer에서 regex 추출 | 없음 |
| Basic/forvo 계열 | Front | 뜻 | 발음 | 예문 / 예문 뜻 |
| sentences | sentence | translation | 없음 | 없음 |

### REQ-004: BGE-M3 임베딩

**WHEN** Document 배치가 준비되면
**THE SYSTEM SHALL** `BAAI/bge-m3` 모델로 Dense 벡터(1024차원)와 Sparse 벡터(SPLADE)를 동시에 생성한다.

**수락 기준**:
- 모델: `BAAI/bge-m3` (FlagEmbedding 라이브러리)
- Dense 벡터: 1024차원 float32
- Sparse 벡터: SPLADE 방식 (token weights dict)
- 배치 처리: 기본 32개씩, 설정 가능
- GPU 자동 감지 (cuda/mps/cpu fallback)
- 임베딩 대상 텍스트: `{word} {meaning} {example}` 조합

### REQ-005: Qdrant 인덱싱

**WHEN** 임베딩이 완료되면
**THE SYSTEM SHALL** Qdrant 컬렉션에 Dense+Sparse 벡터와 메타데이터를 저장한다.

**수락 기준**:
- 컬렉션명: `anki_rag` (설정 가능)
- Dense 벡터: cosine similarity
- Sparse 벡터: Qdrant sparse vector 지원
- 메타데이터 페이로드:
  ```json
  {
    "word": "abandon",
    "meaning": "포기하다, 버리다",
    "pronunciation": "/əˈbændən/",
    "example": "He abandoned the project.",
    "example_translation": "그는 프로젝트를 포기했다.",
    "source": "toefl",
    "deck": "TOEFL 영단어",
    "tags": ["verb", "academic"],
    "note_type": "Simple Model",
    "audio_path": "data/media/hacker_toeic/aban.mp3"  // null if no audio
  }
  ```
- 중복 upsert 지원 (note_id 기반)
- 배치 upsert: 기본 100개씩

### REQ-006: 하이브리드 검색 (RRF Fusion)

**WHEN** 사용자가 검색 쿼리를 입력하면
**THE SYSTEM SHALL** Dense + Sparse 검색 결과를 RRF(Reciprocal Rank Fusion)로 병합하여 상위 K개 결과를 반환한다.

**수락 기준**:
- Dense 검색: cosine similarity top-K
- Sparse 검색: BM25/SPLADE top-K
- RRF 병합: `score = Σ 1/(rank + 60)`
- 기본 top-K: 10
- 필터 지원: `source`, `deck`, `tags` 필드 기반
- 검색 시 쿼리도 BGE-M3로 임베딩

### REQ-007: Claude API RAG 응답 생성

**WHEN** 검색 결과가 준비되면
**THE SYSTEM SHALL** Claude API(claude-sonnet-4-6)를 사용하여 검색된 컨텍스트를 바탕으로 자연어 답변을 생성한다.

**수락 기준**:
- 모델: `claude-sonnet-4-6`
- 시스템 프롬프트: 영어 학습 도우미 역할 정의
- 컨텍스트: 검색된 상위 K개 Document를 구조화하여 프롬프트에 삽입
- 출처 인용: 답변에 source, deck 정보 포함
- 스트리밍 응답 지원 (선택적)
- max_tokens: 1024 (기본값, 설정 가능)

### REQ-008: CLI 인터페이스

**WHEN** 사용자가 CLI를 실행하면
**THE SYSTEM SHALL** 인덱싱, 검색, 대화형 RAG 세 가지 모드를 제공한다.

**수락 기준**:
```bash
# 인덱싱 (오디오 파일 자동 추출 포함)
python -m anki_rag index --data-dir ./data

# 단순 검색
python -m anki_rag search "abandon" --source toefl --top-k 5

# 단순 검색 + 오디오 재생
python -m anki_rag search "abandon" --play-audio

# RAG 질의
python -m anki_rag query "abandon의 뜻과 예문을 알려줘"

# RAG 질의 + 오디오 재생
python -m anki_rag query "abandon의 뜻과 예문을 알려줘" --play-audio

# 대화형 모드 (응답 후 오디오 재생 여부 대화형 선택)
python -m anki_rag chat
```

### REQ-009: 오디오 파일 추출

**WHEN** `.apkg` 파일을 파싱할 때
**THE SYSTEM SHALL** ZIP 내부의 `media` JSON과 번호로 된 미디어 파일을 파싱하여 오디오 파일을 로컬 디렉토리에 추출한다.

**수락 기준**:
- `media` JSON 파싱: `{"0": "aban.mp3", "1": "give_up.mp3", ...}` 형태
- 노트 `flds`에서 `[sound:파일명]` 패턴을 정규식으로 추출하여 해당 노트의 오디오 파일 식별
- 추출 경로: `data/media/{source}/원본파일명.mp3`
- 오디오 없는 덱(toefl, xfer, sentences)은 `audio_path = null` 로 처리
- 지원 형식: `.mp3`, `.ogg`, `.wav` (Anki 지원 형식)
- 이미 추출된 파일은 재추출 스킵 (idempotent)

### REQ-010: 오디오 재생

**WHEN** 사용자가 `--play-audio` 플래그를 사용하거나 대화형 모드에서 오디오 재생을 요청하면
**THE SYSTEM SHALL** 검색 결과의 오디오 파일을 순서대로 재생한다.

**수락 기준**:
- 오디오 파일이 존재하는 결과만 재생 (없으면 무시)
- 재생 순서: 검색 결과 순위 기준 첫 번째 유효 오디오
- 플랫폼별 재생 구현:
  - Windows: `playsound` 라이브러리
  - macOS: `afplay` 서브프로세스
  - Linux: `aplay` 또는 `paplay` 서브프로세스
- 대화형 모드(`chat`): 답변 출력 후 "발음 듣기? [Y/n]" 프롬프트 제공
- 오디오 파일 없거나 재생 실패 시 경고 메시지 출력 후 계속 진행

---

## 3. 비기능 요구사항

### NREQ-001: 성능

- 인덱싱: 24,472개 Document 처리 < 30분 (GPU 환경)
- 검색 레이턴시: < 200ms (Qdrant local)
- RAG 응답: < 5초 (Claude API 포함)

### NREQ-002: 확장성

- 새로운 .apkg 파일 추가 인덱싱 지원 (증분 업데이트)
- source/deck 별 독립 필터링
- 컬렉션 재빌드 없이 새 데이터 upsert

### NREQ-003: 설정 관리

- `config.yaml` 또는 환경변수로 모든 파라미터 설정
- Qdrant 주소, 포트, 컬렉션명
- 임베딩 배치 사이즈
- Claude API 키 (환경변수 `ANTHROPIC_API_KEY`)

---

## 4. 아키텍처

```
data/
├── *.apkg              # Anki 패키지 파일
├── 10000.txt           # 원서 문장
└── media/              # 추출된 오디오 파일
    ├── hacker_toeic/   # source별 디렉토리
    ├── hacker_green/
    └── phrasal/

src/
├── models.py       # Document, SearchResult 데이터 모델 (Pydantic)
├── parser.py       # AnkiParser, TextParser (오디오 파일 추출 포함)
├── embedder.py     # BGE-M3 임베딩 (Dense + Sparse)
├── indexer.py      # Qdrant 인덱서 (upsert, collection 관리)
├── retriever.py    # 하이브리드 검색 (RRF Fusion)
├── audio.py        # 오디오 재생 (플랫폼별 구현)
├── rag.py          # Claude API RAG 파이프라인
└── __main__.py     # CLI 진입점

tests/
├── test_parser.py
├── test_embedder.py
├── test_indexer.py
├── test_retriever.py
├── test_audio.py
└── test_rag.py
```

### 4.1 데이터 흐름

```
.apkg / .txt
    │
    ▼
[parser.py]  ──→  List[Document]
    │
    ▼
[embedder.py]  ──→  List[DocumentWithVectors]
    │                  (dense: List[float], sparse: Dict[int, float])
    ▼
[indexer.py]  ──→  Qdrant Collection (upsert)

    (검색 시)

query (str)
    │
    ▼
[embedder.py]  ──→  query_dense, query_sparse
    │
    ▼
[retriever.py]  ──→  RRF Fusion  ──→  List[SearchResult]
    │
    ▼
[rag.py]  ──→  Claude API  ──→  Answer (str)
```

---

## 5. 기술 의존성

```toml
# pyproject.toml
[tool.poetry.dependencies]
python = "^3.12"
FlagEmbedding = "^1.2"      # BGE-M3 (Dense + Sparse)
qdrant-client = "^1.9"       # Qdrant Python SDK
anthropic = "^0.28"          # Claude API
pydantic = "^2.7"            # 데이터 모델
click = "^8.1"               # CLI
beautifulsoup4 = "^4.12"     # HTML 파싱
torch = "^2.3"               # BGE-M3 백엔드
playsound = "^1.3"           # 오디오 재생 (Windows/macOS)
```

---

## 6. 수락 기준 (Given-When-Then)

### AC-001: TOEFL 단어 파싱

```gherkin
Given toefl_voca_v1.apkg 파일이 data/ 폴더에 존재한다
When AnkiParser.parse("data/toefl_voca_v1.apkg")를 실행하면
Then 5,308개의 Document가 반환된다
And 각 Document는 word, meaning 필드가 비어있지 않다
And source == "toefl"이다
```

### AC-002: 하이브리드 검색

```gherkin
Given Qdrant에 24,472개 Document가 인덱싱되어 있다
When retriever.search("abandon", top_k=5)를 실행하면
Then 5개 이하의 SearchResult가 반환된다
And 첫 번째 결과의 word 필드에 "abandon"이 포함된다
And 각 결과에 score 값이 존재한다
```

### AC-003: RAG 응답

```gherkin
Given Qdrant 검색이 정상 동작한다
And ANTHROPIC_API_KEY 환경변수가 설정되어 있다
When rag.query("abandon의 뜻과 예문을 알려줘")를 실행하면
Then 비어있지 않은 문자열 응답이 반환된다
And 응답에 "abandon"이 언급된다
And 응답에 출처(source 또는 deck) 정보가 포함된다
```

### AC-004: 필터 검색

```gherkin
Given Qdrant에 데이터가 인덱싱되어 있다
When retriever.search("vocabulary", source_filter="toefl", top_k=10)을 실행하면
Then 반환된 모든 결과의 source == "toefl"이다
```

### AC-005: 오디오 파일 추출

```gherkin
Given _-forvo_-youglish_link.apkg 파일이 data/ 폴더에 존재한다
When AnkiParser.parse("data/_-forvo_-youglish_link.apkg")를 실행하면
Then data/media/hacker_toeic/ 디렉토리에 .mp3 파일들이 추출된다
And 오디오가 있는 Document의 audio_path가 null이 아니다
And toefl_voca_v1.apkg에서 파싱된 Document의 audio_path는 null이다
```

### AC-006: 오디오 재생

```gherkin
Given "abandon"으로 검색한 결과에 audio_path가 존재한다
When audio.play(search_result.audio_path)를 실행하면
Then 오디오 파일이 재생된다
And 재생 완료 후 함수가 정상 반환된다

Given audio_path가 null인 SearchResult가 있다
When audio.play(None)을 실행하면
Then 예외 없이 "오디오 없음" 메시지를 출력하고 반환된다
```

---

## 7. 구현 순서

| 단계 | 모듈 | 우선순위 |
|------|------|---------|
| 1 | `src/models.py` | P0 - 모든 모듈의 기반 |
| 2 | `src/parser.py` | P0 - 데이터 파이프라인 시작점 + 오디오 추출 |
| 3 | `src/embedder.py` | P1 - BGE-M3 |
| 4 | `src/indexer.py` | P1 - Qdrant 연동 |
| 5 | `src/retriever.py` | P1 - 하이브리드 검색 |
| 6 | `src/audio.py` | P1 - 오디오 재생 (플랫폼별) |
| 7 | `src/rag.py` | P2 - Claude API |
| 8 | `src/__main__.py` | P2 - CLI (`--play-audio` 플래그 포함) |

---

## 8. 미결 사항 (Open Questions)

모든 미결 사항이 해결되었습니다 (2026-04-03).

- [x] Qdrant 배포 방식: **로컬 파일 기반 유지** (개인 학습용으로 충분)
- [x] 임베딩 캐시: **불필요** (데이터 규모가 작아 재계산 비용 무시 가능)
- [x] 멀티링구얼 쿼리: **현상태 유지** (BGE-M3 다국어 지원, 적중률 99% 검증 완료)
- [x] 청크 전략: **카드 단위 유지** (Anki 카드는 대부분 짧은 텍스트)
- [x] 오디오 여러 개인 경우: **복수 오디오 지원 구현** (audio_paths 리스트, 중복 제거)
- [x] playsound Windows 호환성: **해결됨** (playsound 제거 완료, 웹 UI에서 브라우저 재생)

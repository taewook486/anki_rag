# SPEC-RAG-003: LLM 프로바이더 교체 (ANTHROPIC_API_KEY 제거)

**Status**: Completed
**Created**: 2026-03-20
**Author**: [I
**Parent SPEC**: SPEC-RAG-001
**Tech Stack**: OpenAI SDK (OpenAI-compatible API), Python 3.11+

---

## 1. 개요 (Overview)

현재 RAG 파이프라인이 Anthropic API (ANTHROPIC_API_KEY)에 하드코딩되어 있어, API 키가 없으면 query/chat 기능을 사용할 수 없다.

사용자가 GLM Global Coding Plan (OpenAI 호환 API)을 구독 중이므로, LLM 프로바이더를 OpenAI 호환 API로 교체하고, 향후 다른 프로바이더도 지원할 수 있도록 설계한다.

### 1.1 목표

- ANTHROPIC_API_KEY 의존성 제거
- OpenAI 호환 API (GLM 등) 지원
- 환경변수로 LLM 프로바이더 설정 가능
- Anthropic API도 선택적으로 유지 (향후 API 키 발급 시)

### 1.2 영향 범위

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| src/rag.py | 수정 | Anthropic SDK → OpenAI SDK + 프로바이더 추상화 |
| src/api/routes/query.py | 수정 | RAGPipeline 초기화 변경 |
| src/__main__.py | 수정 | CLI에서 RAGPipeline 초기화 변경 |
| pyproject.toml | 수정 | anthropic → openai 의존성 변경 |
| tests/test_rag.py | 수정 | 테스트 업데이트 |

---

## 2. 기능 요구사항 (EARS Format)

### REQ-001: OpenAI 호환 API 지원

**WHEN** 사용자가 LLM_BASE_URL과 LLM_API_KEY 환경변수를 설정하면
**THE SYSTEM SHALL** 해당 OpenAI 호환 엔드포인트로 LLM 요청을 보낸다.

**수락 기준**:
- `openai` Python SDK 사용
- `LLM_BASE_URL`: API 엔드포인트 (예: `https://api.openai.com/v1`)
- `LLM_API_KEY`: API 인증 키
- `LLM_MODEL`: 사용할 모델명 (기본값: `gpt-4o-mini`)
- Chat Completions API (`/chat/completions`) 형식 사용

### REQ-002: Anthropic API 선택적 지원

**WHEN** ANTHROPIC_API_KEY 환경변수가 설정되어 있으면
**THE SYSTEM SHALL** Anthropic API를 우선 사용한다.

**수락 기준**:
- `ANTHROPIC_API_KEY` 설정 시 기존 Anthropic SDK 사용
- `LLM_API_KEY` 설정 시 OpenAI 호환 API 사용
- 둘 다 없으면 명확한 에러 메시지 출력
- 우선순위: ANTHROPIC_API_KEY > LLM_API_KEY

### REQ-003: 프로바이더 추상화

**THE SYSTEM SHALL** LLM 프로바이더를 추상화하여 교체 가능하게 한다.

**수락 기준**:
- `LLMProvider` 프로토콜/ABC 정의
- `AnthropicProvider`: Anthropic SDK 래퍼
- `OpenAICompatibleProvider`: OpenAI SDK 래퍼 (GLM, OpenAI, OpenRouter 등)
- `RAGPipeline`이 프로바이더 인터페이스에만 의존
- 스트리밍 응답 지원

### REQ-004: 환경변수 설정

**WHEN** 시스템이 시작되면
**THE SYSTEM SHALL** 환경변수에서 LLM 설정을 읽는다.

**수락 기준**:
```
# OpenAI 호환 API (GLM, OpenRouter 등)
LLM_BASE_URL=https://your-glm-endpoint/v1
LLM_API_KEY=your-api-key
LLM_MODEL=gpt-4o-mini

# Anthropic API (선택)
ANTHROPIC_API_KEY=sk-ant-...
```

### REQ-005: pyproject.toml 의존성 변경

**THE SYSTEM SHALL** 핵심 의존성에서 `anthropic`를 제거하고 `openai`를 추가한다.

**수락 기준**:
- 핵심 의존성: `openai>=1.0` 추가
- 선택 의존성: `anthropic>=0.28`을 optional로 이동
```toml
dependencies = [
    "openai>=1.0",      # 추가
    # "anthropic>=0.28", # optional로 이동
]

[project.optional-dependencies]
anthropic = ["anthropic>=0.28"]
```

---

## 3. 설계 (Design)

### 3.1 LLMProvider 프로토콜

```python
from typing import Protocol

class LLMProvider(Protocol):
    def generate(self, prompt: str, stream: bool = False) -> str: ...
    def stream(self, prompt: str) -> Iterator[str]: ...
```

### 3.2 프로바이더 선택 로직

```
ANTHROPIC_API_KEY 존재?
  ├─ Yes → AnthropicProvider
  └─ No → LLM_API_KEY 존재?
        ├─ Yes → OpenAICompatibleProvider(base_url, api_key, model)
        └─ No → ValueError("LLM_API_KEY 또는 ANTHROPIC_API_KEY 필요")
```

### 3.3 RAGPipeline 변경

```python
class RAGPipeline:
    def __init__(self, retriever, provider: LLMProvider | None = None):
        self.retriever = retriever
        self.provider = provider or create_provider()  # 환경변수에서 자동 감지
```

---

## 4. 수락 기준 (Given-When-Then)

### AC-001: OpenAI 호환 API로 RAG 질의

```gherkin
Given LLM_BASE_URL, LLM_API_KEY, LLM_MODEL 환경변수가 설정되어 있다
And ANTHROPIC_API_KEY는 설정되지 않았다
When RAG 질의를 수행하면
Then OpenAI 호환 API를 통해 응답이 생성된다
And 응답이 정상적으로 반환된다
```

### AC-002: Anthropic API 우선 사용

```gherkin
Given ANTHROPIC_API_KEY와 LLM_API_KEY 모두 설정되어 있다
When RAG 질의를 수행하면
Then Anthropic API가 우선 사용된다
```

### AC-003: API 키 미설정 시 에러

```gherkin
Given ANTHROPIC_API_KEY와 LLM_API_KEY 모두 설정되지 않았다
When RAGPipeline을 초기화하면
Then "LLM_API_KEY 또는 ANTHROPIC_API_KEY 환경변수가 필요합니다" 에러가 발생한다
```

### AC-004: 스트리밍 응답

```gherkin
Given LLM_API_KEY가 설정되어 있다
When stream=True로 RAG 질의를 수행하면
Then 응답이 스트리밍으로 출력된다
```

---

## 5. 구현 순서

| 단계 | 작업 | 파일 |
|------|------|------|
| 1 | LLMProvider 프로토콜 + 구현체 작성 | src/rag.py |
| 2 | RAGPipeline 리팩토링 | src/rag.py |
| 3 | pyproject.toml 의존성 변경 | pyproject.toml |
| 4 | CLI 초기화 코드 수정 | src/__main__.py |
| 5 | API 라우트 수정 | src/api/routes/query.py |
| 6 | 테스트 업데이트 | tests/test_rag.py |

---

## 6. 비기능 요구사항

- **호환성**: OpenAI Chat Completions API 호환 엔드포인트 모두 지원
- **성능**: 기존 대비 응답 시간 변화 없음
- **보안**: API 키는 환경변수로만 전달, 코드에 하드코딩 금지

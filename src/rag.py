"""RAG 파이프라인 - 다중 LLM 프로바이더 지원"""

from __future__ import annotations

import logging
import os
from typing import Optional, Iterator, Protocol, runtime_checkable

from openai import OpenAI

from src.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 컨텍스트 품질 상수
# ---------------------------------------------------------------------------
# @MX:NOTE: RRF 점수 임계값 — k=60 기준 단일 리스트 rank≈120에 해당; 빈 컬렉션은 0 반환
_MIN_RRF_SCORE: float = 0.005
_MAX_CONTEXT_CHARS: int = 2000   # 전체 컨텍스트 최대 문자 수
_MAX_EXAMPLE_CHARS: int = 150    # 예문 필드 최대 문자 수

# ---------------------------------------------------------------------------
# 시스템 프롬프트 (Few-shot 예시 2개 포함)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """당신은 영어 학습 전문가입니다.
제공된 [참고 자료]만을 근거로 답변하세요.
참고 자료에 없는 내용은 "검색된 자료에 없습니다"라고 하세요.
단어·뜻·예문·출처(source, deck)를 포함해 답변하세요.

[답변 형식 예시]
질문: abandon의 뜻은?
답변:
**abandon** [/əˈbændən/]
뜻: 포기하다, 버리다
예문: He abandoned the car.
출처: TOEFL 영단어 (toefl)

질문: give up과 비슷한 구동사는?
답변:
give up의 유사 표현:
1. **abandon** — 완전히 버리다 (출처: toefl)
2. **quit** — 그만두다 (출처: phrasal)
공통점: 모두 '중단·포기' 의미를 포함합니다."""


# ---------------------------------------------------------------------------
# LLM Provider Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class LLMProvider(Protocol):
    """LLM 프로바이더 프로토콜"""

    def generate(self, messages: list[dict], model: str, max_tokens: int) -> str:
        """동기 응답 생성"""
        ...

    def stream(self, messages: list[dict], model: str, max_tokens: int) -> Iterator[str]:
        """스트리밍 응답 생성"""
        ...


# ---------------------------------------------------------------------------
# Anthropic 임포트 헬퍼
# ---------------------------------------------------------------------------
def _import_anthropic():
    """anthropic 패키지에서 Anthropic 클래스를 임포트한다.

    Raises:
        ImportError: anthropic 패키지가 설치되지 않았을 때
    """
    try:
        from anthropic import Anthropic
        return Anthropic
    except ImportError:
        raise ImportError(
            "anthropic 패키지가 필요합니다. pip install anki-rag[anthropic]"
        )


def _extract_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """messages 리스트에서 system role 메시지를 분리한다.

    Anthropic API는 system을 최상위 파라미터로 전달해야 하므로
    messages 배열에서 추출하여 분리한다.

    Returns:
        (system_content, non_system_messages)
    """
    system_content = ""
    user_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            system_content = msg.get("content", "")
        else:
            user_messages.append(msg)
    return system_content, user_messages


# ---------------------------------------------------------------------------
# OpenAI 호환 프로바이더
# ---------------------------------------------------------------------------
class OpenAICompatibleProvider:
    """OpenAI 호환 API 프로바이더 (GLM, OpenRouter 등 지원)"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, messages: list[dict], model: str, max_tokens: int) -> str:
        """동기 응답 생성"""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=False,
        )
        return response.choices[0].message.content

    def stream(self, messages: list[dict], model: str, max_tokens: int) -> Iterator[str]:
        """스트리밍 응답 생성"""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content


# ---------------------------------------------------------------------------
# Anthropic 프로바이더
# ---------------------------------------------------------------------------
class AnthropicProvider:
    """Anthropic API 프로바이더 (선택적 의존성)"""

    def __init__(self, api_key: str) -> None:
        Anthropic = _import_anthropic()
        self.client = Anthropic(api_key=api_key)

    def generate(self, messages: list[dict], model: str, max_tokens: int) -> str:
        """동기 응답 생성 — system role을 별도 파라미터로 분리"""
        system_content, user_messages = _extract_system(messages)
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": user_messages,
        }
        if system_content:
            kwargs["system"] = system_content
        message = self.client.messages.create(**kwargs)
        return message.content[0].text

    def stream(self, messages: list[dict], model: str, max_tokens: int) -> Iterator[str]:
        """스트리밍 응답 생성 — system role을 별도 파라미터로 분리"""
        system_content, user_messages = _extract_system(messages)
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": user_messages,
        }
        if system_content:
            kwargs["system"] = system_content
        with self.client.messages.stream(**kwargs) as stream_ctx:
            yield from stream_ctx.text_stream


# ---------------------------------------------------------------------------
# 프로바이더 팩토리
# ---------------------------------------------------------------------------
def create_provider() -> LLMProvider:
    """환경변수 기반으로 LLM 프로바이더를 생성한다.

    우선순위:
        1. ANTHROPIC_API_KEY -> AnthropicProvider
        2. LLM_API_KEY -> OpenAICompatibleProvider
        3. 키 없음 -> ValueError

    Returns:
        LLMProvider 인스턴스

    Raises:
        ValueError: API 키가 설정되지 않았을 때
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    llm_key = os.getenv("LLM_API_KEY")

    # 우선순위 1: Anthropic
    if anthropic_key:
        try:
            return AnthropicProvider(api_key=anthropic_key)
        except ImportError:
            # anthropic 패키지 없으면 폴백
            if llm_key:
                return OpenAICompatibleProvider(
                    api_key=llm_key,
                    base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
                )
            raise

    # 우선순위 2: OpenAI 호환
    if llm_key:
        return OpenAICompatibleProvider(
            api_key=llm_key,
            base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        )

    # 키 없음
    raise ValueError(
        "API 키가 설정되지 않았습니다. "
        "ANTHROPIC_API_KEY 또는 LLM_API_KEY 환경변수를 설정하세요."
    )


# ---------------------------------------------------------------------------
# RAG 파이프라인
# ---------------------------------------------------------------------------
class RAGPipeline:
    """다중 LLM 프로바이더를 지원하는 RAG 파이프라인"""

    def __init__(
        self,
        retriever: HybridRetriever,
        provider: LLMProvider | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
    ):
        """
        Args:
            retriever: HybridRetriever 인스턴스
            provider: LLM 프로바이더 (None이면 create_provider() 사용)
            model: LLM 모델명 (None이면 LLM_MODEL 환경변수 또는 기본값)
            max_tokens: 최대 토큰 수
        """
        self.retriever = retriever
        self.provider = provider or create_provider()
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.max_tokens = max_tokens

    # @MX:ANCHOR: RAG 파이프라인 공개 진입점
    # @MX:REASON: [AUTO] query(), api/routes/query.py, __main__.py, web/app.py 에서 호출
    def query(
        self,
        question: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        """
        질의응답

        Args:
            question: 사용자 질문
            top_k: 검색할 문서 수
            source_filter: source 필터
            stream: 스트리밍 응답 여부

        Returns:
            답변 텍스트
        """
        # 문서 검색
        search_results = self.retriever.search(question, top_k=top_k, source_filter=source_filter)

        # 컨텍스트 구성
        context = self._build_context(search_results)

        # system + user 메시지 구조 (Few-shot은 SYSTEM_PROMPT에 포함)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_content(question, context)},
        ]

        # LLM 호출
        if stream:
            return self._stream_response(messages)
        return self.provider.generate(
            messages=messages,
            model=self.model,
            max_tokens=self.max_tokens,
        )

    def _build_context(self, search_results: list) -> str:
        """검색 결과로 컨텍스트 구성

        - 순위·점수 메타정보 포함
        - _MIN_RRF_SCORE 미만 결과 제외
        - 예문 _MAX_EXAMPLE_CHARS 자 truncation
        - 누적 문자 수 _MAX_CONTEXT_CHARS 초과 시 중단
        """
        context_parts = []
        total_chars = 0

        for result in search_results:
            if result.score < _MIN_RRF_SCORE:
                continue

            doc = result.document
            parts = [
                f"[순위 {result.rank} | 점수 {result.score:.4f}]",
                f"단어: {doc.word}",
                f"뜻: {doc.meaning}",
            ]
            if doc.pronunciation:
                parts.append(f"발음: {doc.pronunciation}")
            if doc.example:
                example = doc.example[:_MAX_EXAMPLE_CHARS]
                if len(doc.example) > _MAX_EXAMPLE_CHARS:
                    example += "..."
                parts.append(f"예문: {example}")
            if doc.example_translation:
                parts.append(f"예문 번역: {doc.example_translation[:100]}")

            source_info = f"출처: {doc.source}"
            if doc.deck:
                source_info += f" ({doc.deck})"
            parts.append(source_info)

            entry = "\n".join(parts)
            total_chars += len(entry)
            if total_chars > _MAX_CONTEXT_CHARS:
                logger.debug("컨텍스트 문자 수 한도 초과 — %d건 포함", len(context_parts))
                break
            context_parts.append(entry)

        return "\n\n---\n\n".join(context_parts)

    def _build_user_content(self, question: str, context: str) -> str:
        """user role에 삽입할 content 구성"""
        if not context.strip():
            return f"[참고 자료]\n(검색 결과 없음)\n\n[질문]\n{question}\n\n답변:"
        return f"[참고 자료]\n{context}\n\n[질문]\n{question}\n\n답변:"

    def _build_prompt(self, question: str, context: str) -> str:
        """하위 호환성 유지용 단일 문자열 프롬프트 구성

        Note: query()는 system role 방식(_build_user_content)을 사용한다.
              이 메서드는 직접 호출 또는 레거시 코드 호환을 위해 유지한다.
        """
        if not context.strip():
            return f"""당신은 영어 학습 도우미입니다. 관련 데이터가 검색되지 않았습니다. 가능한 범위 내에서 질문에 답변해주세요.

질문: {question}

답변:"""
        return f"""당신은 영어 학습 도우미입니다. 다음 컨텍스트를 바탕으로 질문에 답변해주세요.

컨텍스트:
{context}

질문: {question}

답변:"""

    def _stream_response(self, messages: list[dict]) -> str:
        """스트리밍 응답"""
        full_response = ""
        for text in self.provider.stream(
            messages=messages,
            model=self.model,
            max_tokens=self.max_tokens,
        ):
            full_response += text
            logger.debug("stream chunk: %s", text)
        return full_response

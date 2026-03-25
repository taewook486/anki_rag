"""RAG 파이프라인 - 다중 LLM 프로바이더 지원"""

from __future__ import annotations

import os
from typing import Optional, Iterator, Protocol, runtime_checkable

from openai import OpenAI

from src.retriever import HybridRetriever


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
        """동기 응답 생성"""
        message = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )
        return message.content[0].text

    def stream(self, messages: list[dict], model: str, max_tokens: int) -> Iterator[str]:
        """스트리밍 응답 생성"""
        with self.client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        ) as stream_ctx:
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

        # 프롬프트 구성
        prompt = self._build_prompt(question, context)
        messages = [{"role": "user", "content": prompt}]

        # LLM 호출
        if stream:
            return self._stream_response(messages)
        else:
            return self.provider.generate(
                messages=messages,
                model=self.model,
                max_tokens=self.max_tokens,
            )

    def _build_context(self, search_results: list) -> str:
        """검색 결과로 컨텍스트 구성"""
        context_parts = []
        for result in search_results:
            doc = result.document
            parts = [
                f"단어: {doc.word}",
                f"뜻: {doc.meaning}",
            ]
            if doc.pronunciation:
                parts.append(f"발음: {doc.pronunciation}")
            if doc.example:
                parts.append(f"예문: {doc.example}")
            if doc.example_translation:
                parts.append(f"예문 번역: {doc.example_translation}")

            source_info = f"출처: {doc.source}"
            if doc.deck:
                source_info += f" ({doc.deck})"
            parts.append(source_info)

            context_parts.append("\n".join(parts))

        return "\n\n---\n\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """프롬프트 구성"""
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
            print(text, end="", flush=True)
        print()  # 개행
        return full_response

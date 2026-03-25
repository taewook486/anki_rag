"""RAG 파이프라인 테스트 - LLM Provider 추상화 포함"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.models import Document, SearchResult


# ---------------------------------------------------------------------------
# TestOpenAICompatibleProvider
# ---------------------------------------------------------------------------
class TestOpenAICompatibleProvider:
    """OpenAI 호환 프로바이더 테스트"""

    def test_init_from_env(self):
        """Given LLM_API_KEY와 LLM_BASE_URL이 설정되었을 때,
        When OpenAICompatibleProvider를 생성하면,
        Then 올바른 설정으로 초기화된다"""
        from src.rag import OpenAICompatibleProvider

        with patch.dict(
            "os.environ",
            {"LLM_API_KEY": "test-key", "LLM_BASE_URL": "https://api.test.com/v1"},
            clear=False,
        ):
            with patch("src.rag.OpenAI") as mock_openai_cls:
                provider = OpenAICompatibleProvider(
                    api_key="test-key",
                    base_url="https://api.test.com/v1",
                )
                mock_openai_cls.assert_called_once_with(
                    api_key="test-key",
                    base_url="https://api.test.com/v1",
                )

    def test_generate_returns_content(self):
        """Given OpenAI 클라이언트가 정상 응답할 때,
        When generate를 호출하면,
        Then 응답 텍스트가 반환된다"""
        from src.rag import OpenAICompatibleProvider

        with patch("src.rag.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client

            # chat.completions.create 응답 모킹
            mock_choice = MagicMock()
            mock_choice.message.content = "테스트 응답입니다"
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            provider = OpenAICompatibleProvider(api_key="test-key")
            result = provider.generate(
                messages=[{"role": "user", "content": "안녕"}],
                model="gpt-4o-mini",
                max_tokens=100,
            )

            assert result == "테스트 응답입니다"
            mock_client.chat.completions.create.assert_called_once_with(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "안녕"}],
                max_tokens=100,
                stream=False,
            )

    def test_stream_yields_chunks(self):
        """Given OpenAI 클라이언트가 스트리밍 응답할 때,
        When stream을 호출하면,
        Then 텍스트 청크가 순서대로 반환된다"""
        from src.rag import OpenAICompatibleProvider

        with patch("src.rag.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client

            # 스트리밍 응답 모킹
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta.content = "청크1"

            chunk2 = MagicMock()
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta.content = "청크2"

            chunk3 = MagicMock()
            chunk3.choices = [MagicMock()]
            chunk3.choices[0].delta.content = None  # 마지막 청크

            mock_client.chat.completions.create.return_value = iter(
                [chunk1, chunk2, chunk3]
            )

            provider = OpenAICompatibleProvider(api_key="test-key")
            chunks = list(
                provider.stream(
                    messages=[{"role": "user", "content": "안녕"}],
                    model="gpt-4o-mini",
                    max_tokens=100,
                )
            )

            assert chunks == ["청크1", "청크2"]

    def test_default_base_url(self):
        """Given LLM_BASE_URL이 없을 때,
        When OpenAICompatibleProvider를 생성하면,
        Then 기본 OpenAI URL이 사용된다"""
        from src.rag import OpenAICompatibleProvider

        with patch("src.rag.OpenAI") as mock_openai_cls:
            provider = OpenAICompatibleProvider(api_key="test-key")
            mock_openai_cls.assert_called_once_with(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
            )


# ---------------------------------------------------------------------------
# TestAnthropicProvider
# ---------------------------------------------------------------------------
class TestAnthropicProvider:
    """Anthropic 프로바이더 테스트"""

    def test_init_from_api_key(self):
        """Given ANTHROPIC_API_KEY가 설정되었을 때,
        When AnthropicProvider를 생성하면,
        Then Anthropic 클라이언트가 초기화된다"""
        from src.rag import AnthropicProvider

        with patch("src.rag._import_anthropic") as mock_import:
            mock_anthropic_cls = MagicMock()
            mock_import.return_value = mock_anthropic_cls

            provider = AnthropicProvider(api_key="anthropic-test-key")
            mock_anthropic_cls.assert_called_once_with(api_key="anthropic-test-key")

    def test_generate_returns_content(self):
        """Given Anthropic 클라이언트가 정상 응답할 때,
        When generate를 호출하면,
        Then 응답 텍스트가 반환된다"""
        from src.rag import AnthropicProvider

        with patch("src.rag._import_anthropic") as mock_import:
            mock_anthropic_cls = MagicMock()
            mock_import.return_value = mock_anthropic_cls
            mock_client = MagicMock()
            mock_anthropic_cls.return_value = mock_client

            # messages.create 응답 모킹
            mock_content = MagicMock()
            mock_content.text = "Claude 응답입니다"
            mock_message = MagicMock()
            mock_message.content = [mock_content]
            mock_client.messages.create.return_value = mock_message

            provider = AnthropicProvider(api_key="anthropic-test-key")
            result = provider.generate(
                messages=[{"role": "user", "content": "안녕"}],
                model="claude-sonnet-4-6",
                max_tokens=100,
            )

            assert result == "Claude 응답입니다"
            mock_client.messages.create.assert_called_once_with(
                model="claude-sonnet-4-6",
                max_tokens=100,
                messages=[{"role": "user", "content": "안녕"}],
            )

    def test_import_error_raises(self):
        """Given anthropic 패키지가 설치되지 않았을 때,
        When AnthropicProvider를 생성하면,
        Then ImportError가 발생한다"""
        from src.rag import AnthropicProvider

        with patch("src.rag._import_anthropic", side_effect=ImportError(
            "anthropic 패키지가 필요합니다. pip install anki-rag[anthropic]"
        )):
            with pytest.raises(ImportError, match="anthropic 패키지가 필요합니다"):
                AnthropicProvider(api_key="key")

    def test_stream_yields_chunks(self):
        """Given Anthropic 클라이언트가 스트리밍 응답할 때,
        When stream을 호출하면,
        Then 텍스트 청크가 반환된다"""
        from src.rag import AnthropicProvider

        with patch("src.rag._import_anthropic") as mock_import:
            mock_anthropic_cls = MagicMock()
            mock_import.return_value = mock_anthropic_cls
            mock_client = MagicMock()
            mock_anthropic_cls.return_value = mock_client

            # 스트리밍 응답 모킹
            mock_stream_ctx = MagicMock()
            mock_stream = MagicMock()
            mock_stream.text_stream = iter(["청크A", "청크B"])
            mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream)
            mock_stream_ctx.__exit__ = MagicMock(return_value=False)
            mock_client.messages.stream.return_value = mock_stream_ctx

            provider = AnthropicProvider(api_key="anthropic-test-key")
            chunks = list(
                provider.stream(
                    messages=[{"role": "user", "content": "안녕"}],
                    model="claude-sonnet-4-6",
                    max_tokens=100,
                )
            )

            assert chunks == ["청크A", "청크B"]


# ---------------------------------------------------------------------------
# TestCreateProvider
# ---------------------------------------------------------------------------
class TestCreateProvider:
    """create_provider 팩토리 함수 테스트"""

    def test_anthropic_priority(self):
        """Given ANTHROPIC_API_KEY와 LLM_API_KEY가 모두 설정되었을 때,
        When create_provider를 호출하면,
        Then AnthropicProvider가 반환된다"""
        from src.rag import create_provider, AnthropicProvider

        with patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": "anthro-key", "LLM_API_KEY": "llm-key"},
            clear=False,
        ):
            with patch("src.rag._import_anthropic") as mock_import:
                mock_anthropic_cls = MagicMock()
                mock_import.return_value = mock_anthropic_cls

                provider = create_provider()
                assert isinstance(provider, AnthropicProvider)

    def test_openai_fallback(self):
        """Given LLM_API_KEY만 설정되었을 때,
        When create_provider를 호출하면,
        Then OpenAICompatibleProvider가 반환된다"""
        from src.rag import create_provider, OpenAICompatibleProvider

        env = {"LLM_API_KEY": "llm-key", "LLM_BASE_URL": "https://api.test.com/v1"}
        with patch.dict("os.environ", env, clear=False):
            # ANTHROPIC_API_KEY가 없도록 보장
            with patch.dict("os.environ", {}, clear=False):
                import os
                os.environ.pop("ANTHROPIC_API_KEY", None)

                with patch("src.rag.OpenAI"):
                    provider = create_provider()
                    assert isinstance(provider, OpenAICompatibleProvider)

    def test_no_key_raises(self):
        """Given API 키가 전혀 설정되지 않았을 때,
        When create_provider를 호출하면,
        Then ValueError가 발생하며 안내 메시지가 포함된다"""
        from src.rag import create_provider

        with patch.dict(
            "os.environ",
            {},
            clear=True,
        ):
            with pytest.raises(ValueError, match="API 키가 설정되지 않았습니다"):
                create_provider()

    def test_anthropic_import_error_falls_back_to_openai(self):
        """Given ANTHROPIC_API_KEY가 있지만 anthropic 패키지가 없을 때,
        When create_provider를 호출하면,
        Then LLM_API_KEY가 있으면 OpenAICompatibleProvider로 폴백한다"""
        from src.rag import create_provider, OpenAICompatibleProvider

        with patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": "anthro-key", "LLM_API_KEY": "llm-key"},
            clear=False,
        ):
            with patch("src.rag._import_anthropic", side_effect=ImportError):
                with patch("src.rag.OpenAI"):
                    provider = create_provider()
                    assert isinstance(provider, OpenAICompatibleProvider)


# ---------------------------------------------------------------------------
# TestRAGPipeline
# ---------------------------------------------------------------------------
class TestRAGPipeline:
    """RAG 파이프라인 테스트"""

    def _make_search_result(self, word: str = "test", meaning: str = "테스트") -> SearchResult:
        """테스트용 SearchResult 생성 헬퍼"""
        doc = Document(
            word=word,
            meaning=meaning,
            source="toefl",
            deck="TOEFL 영단어",
        )
        return SearchResult(document=doc, score=0.9, rank=1)

    def test_query_returns_answer(self):
        """Given RAG 파이프라인이 초기화되었을 때,
        When 질문을 하면,
        Then 프로바이더를 통해 답변이 반환된다"""
        from src.rag import RAGPipeline

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [self._make_search_result()]

        mock_provider = MagicMock()
        mock_provider.generate.return_value = "abandon은 '버리다'라는 뜻입니다."

        pipeline = RAGPipeline(retriever=mock_retriever, provider=mock_provider)
        result = pipeline.query("abandon 뜻이 뭐야?")

        assert result == "abandon은 '버리다'라는 뜻입니다."
        mock_retriever.search.assert_called_once()
        mock_provider.generate.assert_called_once()

    def test_query_with_source_filter(self):
        """Given source 필터가 있을 때,
        When 질의하면,
        Then 필터가 retriever.search에 전달된다"""
        from src.rag import RAGPipeline

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [self._make_search_result()]

        mock_provider = MagicMock()
        mock_provider.generate.return_value = "답변"

        pipeline = RAGPipeline(retriever=mock_retriever, provider=mock_provider)
        pipeline.query("test", source_filter="toefl")

        mock_retriever.search.assert_called_once_with(
            "test", top_k=5, source_filter="toefl"
        )

    def test_custom_provider_injection(self):
        """Given 커스텀 프로바이더를 주입할 때,
        When RAGPipeline을 생성하면,
        Then 주입된 프로바이더가 사용된다"""
        from src.rag import RAGPipeline

        mock_retriever = MagicMock()
        mock_provider = MagicMock()

        pipeline = RAGPipeline(retriever=mock_retriever, provider=mock_provider)
        assert pipeline.provider is mock_provider

    def test_stream_mode(self, capsys):
        """Given stream=True일 때,
        When query를 호출하면,
        Then 스트리밍 응답이 반환된다"""
        from src.rag import RAGPipeline

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [self._make_search_result()]

        mock_provider = MagicMock()
        mock_provider.stream.return_value = iter(["스트리밍", " 응답"])

        pipeline = RAGPipeline(retriever=mock_retriever, provider=mock_provider)
        result = pipeline.query("test", stream=True)

        assert result == "스트리밍 응답"
        mock_provider.stream.assert_called_once()

    def test_default_provider_auto_created(self):
        """Given provider를 지정하지 않을 때,
        When RAGPipeline을 생성하면,
        Then create_provider()로 자동 생성된다"""
        from src.rag import RAGPipeline

        mock_retriever = MagicMock()

        with patch("src.rag.create_provider") as mock_create:
            mock_create.return_value = MagicMock()
            pipeline = RAGPipeline(retriever=mock_retriever)
            mock_create.assert_called_once()

    def test_model_from_env(self):
        """Given LLM_MODEL 환경변수가 설정되었을 때,
        When RAGPipeline을 생성하면,
        Then 해당 모델명이 사용된다"""
        from src.rag import RAGPipeline

        mock_retriever = MagicMock()
        mock_provider = MagicMock()

        with patch.dict("os.environ", {"LLM_MODEL": "custom-model"}, clear=False):
            pipeline = RAGPipeline(retriever=mock_retriever, provider=mock_provider)
            assert pipeline.model == "custom-model"

    def test_build_context_preserved(self):
        """Given 검색 결과가 있을 때,
        When _build_context를 호출하면,
        Then 기존 형식대로 컨텍스트가 구성된다"""
        from src.rag import RAGPipeline

        mock_retriever = MagicMock()
        mock_provider = MagicMock()

        pipeline = RAGPipeline(retriever=mock_retriever, provider=mock_provider)
        result = self._make_search_result(word="abandon", meaning="버리다")
        context = pipeline._build_context([result])

        assert "단어: abandon" in context
        assert "뜻: 버리다" in context
        assert "출처: toefl" in context

    def test_build_prompt_preserved(self):
        """Given 컨텍스트가 있을 때,
        When _build_prompt를 호출하면,
        Then 기존 형식대로 프롬프트가 구성된다"""
        from src.rag import RAGPipeline

        mock_retriever = MagicMock()
        mock_provider = MagicMock()

        pipeline = RAGPipeline(retriever=mock_retriever, provider=mock_provider)
        prompt = pipeline._build_prompt("abandon 뜻?", "컨텍스트 내용")

        assert "abandon 뜻?" in prompt
        assert "컨텍스트 내용" in prompt
        assert "영어 학습 도우미" in prompt

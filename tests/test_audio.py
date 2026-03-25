"""Tests for audio.py - 오디오 재생"""

import pytest

try:
    from src.audio import AudioPlayer
except ImportError:
    pytest.skip("src.audio not implemented yet", allow_module_level=True)


class TestAudioPlayer:
    """오디오 플레이어 테스트"""

    def test_play_audio_file(self):
        """오디오 파일 재생 테스트"""
        player = AudioPlayer()
        # mock으로 테스트 필요
        # 실제 파일 재생은 CI에서 스킵

    def test_play_none_gracefully(self):
        """None 경로 처리 테스트"""
        player = AudioPlayer()
        # 예외 없이 처리되어야 함
        player.play(None)

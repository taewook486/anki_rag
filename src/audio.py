"""오디오 플레이어 - 플랫폼별 오디오 재생"""

import platform
import subprocess
from typing import Optional
from pathlib import Path


class AudioPlayer:
    """오디오 플레이어"""

    def __init__(self):
        self.system = platform.system()

    def play(self, audio_path: Optional[str]) -> bool:
        """
        오디오 파일 재생

        Args:
            audio_path: 오디오 파일 경로 (None이면 무시)

        Returns:
            재생 성공 여부
        """
        if not audio_path:
            print("오디오 파일이 없습니다.")
            return False

        path = Path(audio_path)
        if not path.exists():
            print(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
            return False

        try:
            if self.system == "Windows":
                return self._play_windows(path)
            elif self.system == "Darwin":  # macOS
                return self._play_macos(path)
            else:  # Linux
                return self._play_linux(path)
        except Exception as e:
            print(f"오디오 재생 실패: {e}")
            return False

    def _play_windows(self, path: Path) -> bool:
        """Windows 오디오 재생 - 시스템 네이티브 PowerShell 사용"""
        # playsound 제거 - Python 3.12+ 호환성 문제
        try:
            # PowerShell MediaPlayer COM 객체 사용
            ps_script = f'$player = New-Object -ComObject WMPlayer.OCX; $player.URL = "{path}"; $player.controls.play()'
            subprocess.run(
                ["powershell", "-Command", ps_script],
                check=True,
                capture_output=True,
                timeout=10,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            # Fallback: Windows 기본 오디오 플레이어 연결
            try:
                subprocess.run(
                    ["cmd", "/c", "start", "", "/min", str(path)],
                    check=True,
                )
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("오디오 플레이어를 찾을 수 없습니다.")
                return False

    def _play_macos(self, path: Path) -> bool:
        """macOS 오디오 재생 (afplay)"""
        subprocess.run(["afplay", str(path)], check=True)
        return True

    def _play_linux(self, path: Path) -> bool:
        """Linux 오디오 재생 (aplay 또는 paplay)"""
        # aplay 시도
        try:
            subprocess.run(["aplay", str(path)], check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # paplay 시도
        try:
            subprocess.run(["paplay", str(path)], check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("오디오 플레이어를 찾을 수 없습니다 (aplay/paplay).")
            return False

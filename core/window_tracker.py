# core/window_tracker.py

import win32gui
import win32con
import ctypes
from typing import Tuple, Optional


class WindowTracker:
    def __init__(self, window_title: str):
        self.window_title = window_title
        self.hwnd = None
        self._set_dpi_aware()

    def _set_dpi_aware(self):
        """
        DPI 스케일링으로 인한 좌표 어긋남 방지
        """
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    def find_window(self) -> Optional[int]:
        """
        롤 클라이언트 창 핸들 탐색
        """
        hwnd = win32gui.FindWindow(None, self.window_title)
        if hwnd == 0:
            return None
        self.hwnd = hwnd
        return hwnd

    def is_window_valid(self) -> bool:
        if self.hwnd is None:
            return False
        return win32gui.IsWindow(self.hwnd)

    def get_window_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """
        반환값: (x, y, width, height)
        """
        if not self.is_window_valid():
            if self.find_window() is None:
                return None

        # 창이 최소화된 경우 제외
        if win32gui.IsIconic(self.hwnd):
            return None

        rect = win32gui.GetWindowRect(self.hwnd)
        x1, y1, x2, y2 = rect

        width = x2 - x1
        height = y2 - y1

        # 비정상 크기 방어
        if width <= 0 or height <= 0:
            return None

        return x1, y1, width, height

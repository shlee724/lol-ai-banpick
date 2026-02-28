from __future__ import annotations
from typing import Optional, Tuple
from PIL import Image

from core.screen_capture import capture_window
from core.window_tracker import WindowTracker

def get_frame(tracker: WindowTracker, sleep_sec: float) -> Tuple[Optional[Image.Image], Optional[Tuple[int,int]]]:
    window_rect = tracker.get_window_rect()
    if window_rect is None or not tracker.hwnd:
        return None, None

    x, y, w, h = window_rect
    frame_img = capture_window(tracker.hwnd, w, h)
    if frame_img is None:
        return None, None

    return frame_img, (w, h)
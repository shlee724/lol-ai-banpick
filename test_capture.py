from core.window_tracker import WindowTracker
from core.screen_capture import capture_window
import time

tracker = WindowTracker("League of Legends")

while True:
    rect = tracker.get_window_rect()
    if rect and tracker.hwnd:
        x, y, w, h = rect
        img = capture_window(tracker.hwnd, w, h)
        img.save("test_capture.png")
        print("롤 클라이언트 캡처 성공")
        break
    else:
        print("롤 클라이언트 없음")
        time.sleep(1)

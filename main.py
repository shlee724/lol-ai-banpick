from core.window_tracker import WindowTracker
import time

tracker = WindowTracker("League of Legends")

while True:
    rect = tracker.get_window_rect()
    if rect is None:
        print("롤 클라이언트 찾을 수 없음")
    else:
        x, y, w, h = rect
        print(f"창 위치: ({x},{y}) 크기: {w}x{h}")

    time.sleep(0.3)

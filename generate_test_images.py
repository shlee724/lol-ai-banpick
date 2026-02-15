from core.window_tracker import WindowTracker
from core.screen_capture import capture_window
from core.roi_manager import crop_roi_relative_xy
from config.roi import ROI
from config.path import PATHS
import time

tracker = WindowTracker("League of Legends")

while True:
    timestamp = int(time.time() * 1000)

    rect = tracker.get_window_rect()
    if rect is None:
        print("롤 클라이언트 찾을 수 없음")

    elif rect and tracker.hwnd:
        x, y, w, h = rect
        print(f"창 위치: ({x},{y}) 크기: {w}x{h}")

        # 전체 화면 캡처
        img = capture_window(tracker.hwnd, w, h)

        lol_path = PATHS.TEST_LOL_CLIENT_DIR / f"lol_client_{timestamp}.png"
        img.save(lol_path)

        # ROI 캡처
        roi_img = crop_roi_relative_xy(img, rect, ROI.BANPICK_STATUS_TEXT)

        banpick_path = PATHS.TEST_BANPICK_STATUS_DIR / f"banpick_status_{timestamp}.png"
        roi_img.save(banpick_path)

        print("테스트 이미지 생성:", timestamp)

    time.sleep(1)

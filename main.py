from core.window_tracker import WindowTracker
from core.screen_capture import capture_window
from core.roi_manager import crop_roi_definite_xy, crop_roi_relative_xy
from config.roi import ROI
import time

tracker = WindowTracker("League of Legends")

while True:
    rect = tracker.get_window_rect()
    if rect is None:
        print("롤 클라이언트 찾을 수 없음")
    elif rect and tracker.hwnd:
        x, y, w, h = rect
        print(f"창 위치: ({x},{y}) 크기: {w}x{h}")
        img = capture_window(tracker.hwnd, w, h)        #롤 클라이언트 전체 이미지 (Image.Image)
        img.save("captured_images/test_capture.png")

        #roi_img = crop_roi_definite_xy(img, 0, 0, 800, 450)
        roi_img = crop_roi_relative_xy(img, rect ,ROI["banpick_status_text"])
        roi_img.save("captured_images/test_roi_capture.png")

        print("롤 클라이언트 캡처 성공")

    time.sleep(0.3)

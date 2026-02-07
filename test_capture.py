from core.window_tracker import WindowTracker
from core.screen_capture import capture_window
from core.roi_manager import crop_roi_definite_xy, crop_roi_relative_xy
from config.roi import ROI
import time

tracker = WindowTracker("League of Legends")

while True:
    rect = tracker.get_window_rect()
    if rect and tracker.hwnd:
        x, y, w, h = rect
        img = capture_window(tracker.hwnd, w, h)
        img.save("captured_images/test_capture.png")

        #roi_img = crop_roi_definite_xy(img, 0, 0, 800, 450)
        roi_img = crop_roi_relative_xy(img, rect ,ROI["banpick_status_text"])
        roi_img.save("captured_images/test_roi_capture.png")

        print("롤 클라이언트 캡처 성공")
        break
    else:
        print("롤 클라이언트 없음")
        time.sleep(1)

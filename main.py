from core.window_tracker import WindowTracker
from core.screen_capture import capture_window
from core.roi_manager import crop_roi_definite_xy, crop_roi_relative_xy
from config.roi import ROI
from config.path import PATHS
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
        img.save(PATHS["LOL_CLIENT_CAPTURE"])

        roi_img = crop_roi_relative_xy(img, rect ,ROI["banpick_status_text"])   
        roi_img.save(PATHS["BANPICK_STATUS_TEXT_CAPTURE"])

        print("롤 클라이언트 캡처 성공")

    time.sleep(0.3)

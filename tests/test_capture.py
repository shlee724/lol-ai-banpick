from core.window_tracker import WindowTracker
from core.screen_capture import capture_window
from core.roi_manager import crop_roi_definite_xy, crop_roi_relative_xy
from config.roi import ROI
from config.path import PATHS
from core.ocr_engine import extract_text
import time

tracker = WindowTracker("League of Legends")

while True:
    rect = tracker.get_window_rect()
    if rect and tracker.hwnd:
        x, y, w, h = rect
        img = capture_window(tracker.hwnd, w, h)
        img.save(PATHS.LOL_CLIENT_CAPTURE)

        roi_img = crop_roi_relative_xy(img, rect ,ROI["banpick_status_text"])
        roi_img.save(PATHS.BANPICK_STATUS_TEXT_CAPTURE)

        text = extract_text(roi_img)
        print("OCR 결과:", text)

        print("롤 클라이언트 캡처 성공")
        break
    else:
        print("롤 클라이언트 없음")
        time.sleep(1)

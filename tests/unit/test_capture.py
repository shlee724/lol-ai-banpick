import time

from config.path import PATHS
from config.roi import ROI
from core.ocr_engine import extract_text
from core.roi_manager import crop_roi_relative_xy
from core.screen_capture import capture_window
from core.window_tracker import WindowTracker

tracker = WindowTracker("League of Legends")

while True:
    rect = tracker.get_window_rect()
    if rect and tracker.hwnd:
        x, y, w, h = rect
        window_size = (w, h)
        img = capture_window(tracker.hwnd, w, h)
        img.save(PATHS.LOL_CLIENT_CAPTURE_PNG)

        roi_img = crop_roi_relative_xy(img, window_size, ROI.BANPICK_STATUS_TEXT)
        roi_img.save(PATHS.BANPICK_STATUS_TEXT_CAPTURE_PNG)

        text = extract_text(roi_img)
        print("OCR 결과:", text)

        print("롤 클라이언트 캡처 성공")
        break
    else:
        print("롤 클라이언트 없음")
        time.sleep(1)

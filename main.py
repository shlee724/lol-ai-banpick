from core.window_tracker import WindowTracker
from core.screen_capture import capture_window
from core.roi_manager import crop_roi_definite_xy, crop_roi_relative_xy
from config.roi import ROI
from config.path import PATHS
from pipeline.normalizer import TextNormalizer
from pipeline.classifier import StateClassifier
from pipeline.buffer import StateBuffer
from pipeline.state_manager import StableStateManager
from pipeline.pick_stage_detector import detect_pick_kind_from_banned_strips
from core.ocr_engine import extract_text
import time

tracker = WindowTracker("League of Legends")
normalizer = TextNormalizer()
classifier = StateClassifier()
buffer = StateBuffer(size=7)
state_manager = StableStateManager(
    min_duration=1.0,
    min_confidence=0.7
)

while True:
    rect = tracker.get_window_rect()
    if rect is None:
        print("롤 클라이언트 찾을 수 없음")
    elif rect and tracker.hwnd:
        x, y, w, h = rect
        print(f"창 위치: ({x},{y}) 크기: {w}x{h}")
        img = capture_window(tracker.hwnd, w, h)        #롤 클라이언트 전체 이미지 (Image.Image)
        img.save(PATHS["LOL_CLIENT_CAPTURE"])

        status_img = crop_roi_relative_xy(img, rect ,ROI["banpick_status_text"])   #밴픽 상태메시지 캡처
        status_img.save(PATHS["BANPICK_STATUS_TEXT_CAPTURE"])
        my_banned = crop_roi_relative_xy(img, rect, ROI["banned_champions_area_my_team"])
        enemy_banned = crop_roi_relative_xy(img, rect, ROI["banned_champions_area_enemy_team"])

        # OCR
        text = extract_text(status_img)

        # Pipeline
        norm = normalizer.normalize(text)
        state = classifier.classify(norm)

        buffer.push(state)
        candidate = buffer.get_majority()
        confidence = buffer.get_confidence()

        stable_state = state_manager.update(candidate, confidence)       
        print(f" StableState → {stable_state}") 

        if stable_state == "PICK":
            pick_res = detect_pick_kind_from_banned_strips(my_banned, enemy_banned, std_threshold=25.0)
            print("PICK 판정:", pick_res.kind, "std:", round(pick_res.std, 2))

            if pick_res.kind == "PICK_REAL":
                # 진짜 픽 단계 로직 실행
                pass

    time.sleep(0.3)

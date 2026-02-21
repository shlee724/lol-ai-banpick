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
from core.gemini_vision import analyze_image_json
from config.prompts import PICKED_CHAMPS_WITH_ROLES_PROMPT, BANNED_CHAMPS_10_PROMPT, DRAFT_FROM_IMAGE_PROMPT_LITE
from config.prompts import build_draft_recommend_prompt, build_draft_recommend_prompt_lite
from core.draft_schema import normalize_picks_with_roles, normalize_bans10, safe_get_draft_fields
from core.gemini_text import generate_text_json
from PIL import Image
import time
import json

tracker = WindowTracker("League of Legends")
normalizer = TextNormalizer()
classifier = StateClassifier()
buffer = StateBuffer(size=7)
state_manager = StableStateManager(
    min_duration=1.0,
    min_confidence=0.7
)

SLEEP_SEC = 0.1
STD_THRESHOLD = 30.0
MODEL_VISION = "gemini-3-flash-preview"
MODEL_TEXT = "gemini-3-flash-preview"

MY_ROLE = "MID"   # TOP/JUNGLE/MID/ADC/SUPPORT 중 하나로 고정
MY_TIER = "BRONZE"     # UNRANKED/IRON/BRONZE/SILVER/GOLD/PLATINUM/EMERALD/DIAMOND/MASTER/GRANDMASTER/CHALLENGER
MY_CHAMP_POOL = ["Malzahar", "Oriana", "Galio", "Mundo", "Garen", "Malphite", "Cho'gath", "Nasus", "kassadin"]  # 예시

DEBUG_SAVE = False

def merge_images_horizontal(img1: Image.Image, img2: Image.Image, bg_color=(255, 255, 255)) -> Image.Image:
    new_width = img1.width + img2.width
    new_height = max(img1.height, img2.height)

    new_img = Image.new("RGB", (new_width, new_height), bg_color)
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    return new_img

while True:
    window_rect_screen = tracker.get_window_rect()
    if window_rect_screen is None:
        print("롤 클라이언트 찾을 수 없음")
    elif window_rect_screen and tracker.hwnd:
        x, y, w, h = window_rect_screen
        window_size = (w, h)
        print(f"창 위치: ({x},{y}) 크기: {w}x{h}")
        img = capture_window(tracker.hwnd, w, h)        #롤 클라이언트 전체 이미지 (Image.Image)
        status_img = crop_roi_relative_xy(img, window_size ,ROI.BANPICK_STATUS_TEXT)   #밴픽 상태메시지 캡처

        if DEBUG_SAVE:
            img.save(PATHS.LOL_CLIENT_CAPTURE_PNG)
            status_img.save(PATHS.BANPICK_STATUS_TEXT_CAPTURE_PNG)

        my_banned_img = crop_roi_relative_xy(img, window_size, ROI.BANNED_CHAMPIONS_MY_TEAM)
        enemy_banned_img = crop_roi_relative_xy(img, window_size, ROI.BANNED_CHAMPIONS_ENEMY_TEAM)
        my_picked_img = crop_roi_relative_xy(img, window_size, ROI.PICKED_CHAMPIONS_MY_TEAM)
        enemy_picked_img = crop_roi_relative_xy(img, window_size, ROI.PICKED_CHAMPIONS_ENEMY_TEAM)
        total_banned_img = merge_images_horizontal(my_banned_img, enemy_banned_img)
        total_picked_img = merge_images_horizontal(my_picked_img, enemy_picked_img)

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
            pick_res = detect_pick_kind_from_banned_strips(my_banned_img, enemy_banned_img, std_threshold=STD_THRESHOLD)
            print("PICK 판정:", pick_res.kind, "std:", round(pick_res.std, 2))

            if pick_res.kind == "PICK_REAL":
                # 진짜 픽 단계 로직 실행
                # 제미나이 api에 픽 정보 보내기
                prompt = DRAFT_FROM_IMAGE_PROMPT_LITE.format(
                    my_role=MY_ROLE,
                    my_tier=MY_TIER,
                    pool_json=json.dumps(MY_CHAMP_POOL, ensure_ascii=False),
                )
                res = analyze_image_json(total_picked_img, prompt=prompt, model=MODEL_VISION)

                my_team, enemy_team, reco, err = safe_get_draft_fields(res)
                if err:
                    print(" ❌ 실패:", p.name, "|", err.get("_error"))
                    if err.get("_error") == "missing_keys":
                        print("   실제키:", err.get("_keys"))
                        try:
                            print("   원문(앞부분):", json.dumps(err.get("_raw"), ensure_ascii=False)[:500])
                        except Exception:
                            print("   원문:", err.get("_raw"))
                    elif err.get("_error") == "json_parse_failed":
                        raw = err.get("_raw", "")
                        print("   raw(앞부분):", raw[:300] if isinstance(raw, str) else raw)
                    else:
                        print("   raw:", err.get("_raw"))
                    continue

                print(" ✅ my_team:", my_team)
                print(" ✅ enemy_team:", enemy_team)
                print(" ✅ reco:", reco)

                break

        if stable_state == "PREPARE":
            continue

    time.sleep(SLEEP_SEC)

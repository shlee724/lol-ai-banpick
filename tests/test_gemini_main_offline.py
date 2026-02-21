from pathlib import Path
from PIL import Image
import time
import json

from config.roi import ROI
from config.path import PATHS

from core.roi_manager import crop_roi_relative_xy
from core.ocr_engine import extract_text
from core.gemini_vision import analyze_image_json
from core.draft_schema import safe_get_draft_fields

from pipeline.normalizer import TextNormalizer
from pipeline.classifier import StateClassifier
from pipeline.buffer import StateBuffer
from pipeline.state_manager import StableStateManager
from pipeline.pick_stage_detector import detect_pick_kind_from_banned_strips


from config.prompts import DRAFT_FROM_IMAGE_PROMPT_LITE

# ======================
# 테스트 설정
# ======================
MY_ROLE = "MID"   # TOP/JUNGLE/MID/ADC/SUPPORT
MY_TIER = "BRONZE"
MY_CHAMP_POOL = ["Malzahar", "Oriana", "Galio", "Mundo", "Garen", "Malphite", "Cho'gath", "Nasus", "kassadin"]

MODEL_VISION = "gemini-3-flash-preview"

SLEEP_SEC = 0.0              # 오프라인이니 0 가능
STD_THRESHOLD = 30.0         # 밴 영역 std로 PICK_REAL 판정 임계값
gemini_cooldown_sec = 1.5    # 연속 호출 방지
max_gemini_calls = 5         # 안전장치

# ======================
# 유틸
# ======================
def merge_images_horizontal(img1: Image.Image, img2: Image.Image, bg_color=(255, 255, 255)) -> Image.Image:
    new_width = img1.width + img2.width
    new_height = max(img1.height, img2.height)

    new_img = Image.new("RGB", (new_width, new_height), bg_color)
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    return new_img

# ======================
# 메인 테스트 루프
# ======================
def main():
    img_dir = PATHS.TEST_LOL_CLIENT_DIR
    paths = sorted(img_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"테스트 이미지가 없음: {img_dir}")

    normalizer = TextNormalizer()
    classifier = StateClassifier()
    buffer = StateBuffer(size=7)
    state_manager = StableStateManager(min_duration=1.0, min_confidence=0.7)

    stable_state = "UNKNOWN"
    last_gemini_call_t = 0.0
    gemini_calls = 0

    for idx, p in enumerate(paths, start=1):
        img = Image.open(p).convert("RGB")
        window_size = (img.width, img.height)

        status_img = crop_roi_relative_xy(img, window_size, ROI.BANPICK_STATUS_TEXT)

        my_banned_img = crop_roi_relative_xy(img, window_size, ROI.BANNED_CHAMPIONS_MY_TEAM)
        enemy_banned_img = crop_roi_relative_xy(img, window_size, ROI.BANNED_CHAMPIONS_ENEMY_TEAM)

        my_picked_img = crop_roi_relative_xy(img, window_size, ROI.PICKED_CHAMPIONS_MY_TEAM)
        enemy_picked_img = crop_roi_relative_xy(img, window_size, ROI.PICKED_CHAMPIONS_ENEMY_TEAM)

        total_picked_img = merge_images_horizontal(my_picked_img, enemy_picked_img)

        # OCR → 상태 분류
        ocr = extract_text(status_img)
        norm = normalizer.normalize(ocr)
        cls = classifier.classify(norm)

        buffer.push(cls)
        candidate = buffer.get_majority()
        confidence = buffer.get_confidence()
        stable_state = state_manager.update(candidate, confidence)

        print("-" * 70)
        print(f"#{idx:04d} 🖼 {p.name}")
        print(f" StableState → {stable_state} | OCR={ocr!r} | norm={norm!r} | cls={cls!r} | buf={candidate}({confidence:.2f})")

        if stable_state == "PICK":
            pick_res = detect_pick_kind_from_banned_strips(my_banned_img, enemy_banned_img, std_threshold=STD_THRESHOLD)
            print(" PICK 판정:", pick_res.kind, "std:", round(pick_res.std, 2))

            if pick_res.kind != "PICK_REAL":
                continue

            # Gemini 호출 쿨다운 + 호출 횟수 제한
            now = time.time()
            if now - last_gemini_call_t < gemini_cooldown_sec:
                print(" (Gemini cooldown)")
                continue
            if gemini_calls >= max_gemini_calls:
                print(" (Gemini max calls reached)")
                break

            prompt = DRAFT_FROM_IMAGE_PROMPT_LITE.format(
                my_role=MY_ROLE,
                my_tier=MY_TIER,
                pool_json=json.dumps(MY_CHAMP_POOL, ensure_ascii=False),
            )

            try:
                res = analyze_image_json(total_picked_img, prompt=prompt, model=MODEL_VISION)
            except Exception as e:
                print(" ❌ Gemini 호출 실패:", repr(e))
                continue

            gemini_calls += 1
            last_gemini_call_t = now

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
            # main.py와 동일하게 PREPARE는 다음 프레임
            print(" (PREPARE → 다음 프레임)")
            continue

        if SLEEP_SEC:
            time.sleep(SLEEP_SEC)


if __name__ == "__main__":
    main()

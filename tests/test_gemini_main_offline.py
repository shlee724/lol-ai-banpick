from pathlib import Path
from PIL import Image
import time
import re

from config.roi import ROI
from config.path import PATHS

from core.roi_manager import crop_roi_relative_xy
from core.ocr_engine import extract_text

from pipeline.normalizer import TextNormalizer
from pipeline.classifier import StateClassifier
from pipeline.buffer import StateBuffer
from pipeline.state_manager import StableStateManager
from pipeline.pick_stage_detector import detect_pick_kind_from_banned_strips

from core.gemini_vision import analyze_image_json
from config.prompts import PICKED_CHAMPS_WITH_ROLES_PROMPT, BANNED_CHAMPS_10_PROMPT
from core.draft_schema import normalize_bans10, normalize_picks_with_roles
from config.prompts import build_draft_recommend_prompt
from core.gemini_text import generate_text_json

MY_ROLE = "MID"   # TOP/JUNGLE/MID/ADC/SUPPORT 중 하나로 고정
MY_TIER = "BRONZE"     # UNRANKED/IRON/BRONZE/SILVER/GOLD/PLATINUM/EMERALD/DIAMOND/MASTER/GRANDMASTER/CHALLENGER
MY_CHAMP_POOL = ["Malzahar", "Oriana", "Galio", "Mundo", "Garen"]  # 예시

TS_PATTERN = re.compile(r".*_(\d{10,})\.(png|jpg|jpeg)$", re.IGNORECASE)


def merge_images_horizontal(img1: Image.Image, img2: Image.Image, bg_color=(255, 255, 255)) -> Image.Image:
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")

    new_width = img1.width + img2.width
    new_height = max(img1.height, img2.height)

    new_img = Image.new("RGB", (new_width, new_height), bg_color)
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    return new_img


def _extract_ts_ms(filename: str) -> int | None:
    m = TS_PATTERN.match(filename)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def run_offline_gemini_test(
    *,
    std_threshold: float = 25.0,
    model: str = "gemini-2.0-flash",
    buffer_size: int = 7,
    min_duration: float = 1.0,
    min_confidence: float = 0.7,
    simulate_timing: bool = True,
    max_sleep_sec: float = 0.5,
    gemini_cooldown_sec: float = 2.0,
    max_gemini_calls: int = 20,
):
    """
    main.py 로직을 lol_client 테스트 이미지 순회로 재현 + PICK_REAL에서 Gemini 호출.

    - simulate_timing=True면 파일명 timestamp(ms) 기준으로 sleep을 흉내내서
      StableStateManager(min_duration)이 실제처럼 동작하게 함.
    - gemini_cooldown_sec: Gemini 호출을 너무 자주 하지 않게 쿨다운.
    - max_gemini_calls: 테스트 중 호출 상한(요금 폭탄 방지).
    """
    img_dir: Path = PATHS.GEN_TEST_LOL_CLIENT_CAPTURE
    if not img_dir.exists():
        print("❌ lol_client 테스트 폴더 없음:", img_dir)
        return

    img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    print(f"📂 lol_client 이미지 수: {len(img_files)}")
    print(
        f"설정: buffer={buffer_size}, min_duration={min_duration}, min_conf={min_confidence}, "
        f"std_th={std_threshold}, model={model}, cooldown={gemini_cooldown_sec}s\n"
    )

    normalizer = TextNormalizer()
    classifier = StateClassifier()
    buffer = StateBuffer(size=buffer_size)
    state_manager = StableStateManager(min_duration=min_duration, min_confidence=min_confidence)

    prev_ts_ms = None
    last_gemini_call_ts = 0.0
    gemini_calls = 0

    # 같은 결과 반복 호출 방지용(간단 캐시)
    last_sent_signature = None

    for idx, img_path in enumerate(img_files):
        try:
            img = Image.open(img_path)
            window_size = (img.width, img.height)  # 테스트에서는 이미지 전체를 윈도우로 가정

            # === ROI crop (main.py와 동일한 흐름) ===
            status_img = crop_roi_relative_xy(img, window_size, ROI["banpick_status_text"])

            my_banned_img = crop_roi_relative_xy(img, window_size, ROI["banned_champions_area_my_team"])
            enemy_banned_img = crop_roi_relative_xy(img, window_size, ROI["banned_champions_area_enemy_team"])
            total_banned_img = merge_images_horizontal(my_banned_img, enemy_banned_img)

            my_picked_img = crop_roi_relative_xy(img, window_size, ROI["picked_champions_area_my_team"])
            enemy_picked_img = crop_roi_relative_xy(img, window_size, ROI["picked_champions_area_enemy_team"])
            total_picked_img = merge_images_horizontal(my_picked_img, enemy_picked_img)

            # === OCR + pipeline ===
            text = extract_text(status_img)
            norm = normalizer.normalize(text)
            state = classifier.classify(norm)

            buffer.push(state)
            candidate = buffer.get_majority()
            confidence = buffer.get_confidence()

            stable_state = state_manager.update(candidate, confidence)

            print(f"#{idx:04d} 🖼 {img_path.name}")
            print(f" StableState → {stable_state} | OCR='{text}' | norm='{norm}' | cls='{state}' | buf={candidate}({confidence:.2f})")

            if stable_state == "PICK":
                pick_res = detect_pick_kind_from_banned_strips(
                    my_banned_img, enemy_banned_img, std_threshold=std_threshold
                )
                print(" PICK 판정:", pick_res.kind, "std:", round(pick_res.std, 2))

                if pick_res.kind == "PICK_REAL":
                    # 진짜 픽 단계 로직 실행
                    # 제미나이 api에 픽 정보 보내기
                    raw = analyze_image_json(total_picked_img, prompt=PICKED_CHAMPS_WITH_ROLES_PROMPT, model="gemini-2.5-flash")
                    picked = normalize_picks_with_roles(raw)
                    print(picked.my_team)     # {"top": "...", "jungle": "...", ...}
                    print(picked.enemy_team)  # [..5..]     

                    # 제미나이 api에 밴 정보 보내기
                    raw = analyze_image_json(total_banned_img, prompt=BANNED_CHAMPS_10_PROMPT, model="gemini-2.5-flash")
                    bans10 = normalize_bans10(raw)
                    print(bans10.bans)
                    
                    # 제미나이 api에 밴픽 추천
                    prompt = build_draft_recommend_prompt(
                        my_role=MY_ROLE,
                        my_tier=MY_TIER,
                        my_champ_pool=MY_CHAMP_POOL,
                        my_team=picked.my_team,
                        enemy_picks=picked.enemy_team,
                        bans_10=bans10.bans,
                    )

                    rec = generate_text_json(prompt, model="gemini-2.5-flash")
                    print("📌 추천:", rec)

                    break

            print("-" * 70)

            # === timing simulation ===
            if simulate_timing:
                ts_ms = _extract_ts_ms(img_path.name)
                if ts_ms is not None and prev_ts_ms is not None:
                    dt = max(0.0, (ts_ms - prev_ts_ms) / 1000.0)
                    time.sleep(min(dt, max_sleep_sec))
                else:
                    time.sleep(0.3)

                if ts_ms is not None:
                    prev_ts_ms = ts_ms
            else:
                time.sleep(0.01)

        except Exception as e:
            print(f"❌ 실패: {img_path.name} | {e}")
            print("-" * 70)


if __name__ == "__main__":
    run_offline_gemini_test(
        std_threshold=25.0,
        model="gemini-2.5-flash",
        buffer_size=7,
        min_duration=1.0,
        min_confidence=0.7,
        simulate_timing=True,
        max_sleep_sec=0.5,
        gemini_cooldown_sec=2.0,
        max_gemini_calls=20,
    )

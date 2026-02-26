import time
from typing import List

from PIL import Image

from config.path import PATHS
from config.roi import ROI
from core.lol_pick_coach import get_client, lol_mid_pick_coach_stream
from core.lol_playplan_coach import get_playplan_coach_client, lol_playplan_stream
from core.ocr_engine import extract_text
from core.roi_manager import crop_roi_relative_xy
from pipeline.buffer import StateBuffer
from pipeline.classifier import StateClassifier
from pipeline.normalizer import TextNormalizer
from pipeline.pick_stage_detector import detect_pick_kind_from_banned_strips
from pipeline.prepare_phase_detector import is_dual_timer_effective
from pipeline.state_manager import StableStateManager

# ======================
# 테스트 설정
# ======================
MY_ROLE = "MID"  # TOP/JUNGLE/MID/ADC/SUPPORT
MY_TIER = "BRONZE"
MY_CHAMP_POOL = [
    "Malzahar",
    "Oriana",
    "Galio",
    "Mundo",
    "Garen",
    "Malphite",
    "Cho'gath",
    "Nasus",
    "kassadin",
]

MODEL_VISION = "gemini-2.5-flash"

SLEEP_SEC = 0.0  # 오프라인이니 0 가능
STD_THRESHOLD = 30.0  # 밴 영역 std로 PICK_REAL 판정 임계값
gemini_cooldown_sec = 1.5  # 연속 호출 방지
max_gemini_calls = 5  # 안전장치


# ======================
# 유틸
# ======================
def merge_images_horizontal(
    img1: Image.Image, img2: Image.Image, bg_color=(255, 255, 255)
) -> Image.Image:
    new_width = img1.width + img2.width
    new_height = max(img1.height, img2.height)

    new_img = Image.new("RGB", (new_width, new_height), bg_color)
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    return new_img


def run_streaming(label: str, stream_iter) -> str:
    buf = []
    t0 = time.perf_counter()
    first_token_t = None

    for delta in stream_iter:
        if first_token_t is None:
            first_token_t = time.perf_counter()
            print(f"\n[{label}] ⏱ 첫 토큰: {first_token_t - t0:.2f}s\n")
        print(delta, end="", flush=True)
        buf.append(delta)

    t1 = time.perf_counter()
    print(f"\n\n[{label}] ⏱ 전체: {t1 - t0:.2f}s")
    return "".join(buf)


def merge_images_vertical(images: List[Image.Image], bg_color=(255, 255, 255)) -> Image.Image:
    """
    여러 이미지를 세로로 이어 붙인다.
    - 너비는 가장 넓은 이미지 기준
    - 빈 공간은 bg_color로 채움
    """

    if not images:
        raise ValueError("이미지 리스트가 비어 있음")

    new_width = max(img.width for img in images)
    new_height = sum(img.height for img in images)

    new_img = Image.new("RGB", (new_width, new_height), bg_color)

    y_offset = 0
    for img in images:
        new_img.paste(img, (0, y_offset))
        y_offset += img.height

    return new_img


def crop_picked_champs_texts_area(img: Image.Image, window_size: tuple[int, int]):
    my_picked_1 = crop_roi_relative_xy(img, window_size, ROI.MY_TEAM_PICK1)
    my_picked_2 = crop_roi_relative_xy(img, window_size, ROI.MY_TEAM_PICK2)
    my_picked_3 = crop_roi_relative_xy(img, window_size, ROI.MY_TEAM_PICK3)
    my_picked_4 = crop_roi_relative_xy(img, window_size, ROI.MY_TEAM_PICK4)
    my_picked_5 = crop_roi_relative_xy(img, window_size, ROI.MY_TEAM_PICK5)
    enemy_picked_1 = crop_roi_relative_xy(img, window_size, ROI.ENEMY_TEAM_PICK1)
    enemy_picked_2 = crop_roi_relative_xy(img, window_size, ROI.ENEMY_TEAM_PICK2)
    enemy_picked_3 = crop_roi_relative_xy(img, window_size, ROI.ENEMY_TEAM_PICK3)
    enemy_picked_4 = crop_roi_relative_xy(img, window_size, ROI.ENEMY_TEAM_PICK4)
    enemy_picked_5 = crop_roi_relative_xy(img, window_size, ROI.ENEMY_TEAM_PICK5)
    my_pos_1 = crop_roi_relative_xy(img, window_size, ROI.MY_TEAM_POS1)
    my_pos_2 = crop_roi_relative_xy(img, window_size, ROI.MY_TEAM_POS2)
    my_pos_3 = crop_roi_relative_xy(img, window_size, ROI.MY_TEAM_POS3)
    my_pos_4 = crop_roi_relative_xy(img, window_size, ROI.MY_TEAM_POS4)
    my_pos_5 = crop_roi_relative_xy(img, window_size, ROI.MY_TEAM_POS5)

    my_list = [
        my_pos_1,
        my_picked_1,
        my_pos_2,
        my_picked_2,
        my_pos_3,
        my_picked_3,
        my_pos_4,
        my_picked_4,
        my_pos_5,
        my_picked_5,
    ]
    enemy_list = [enemy_picked_1, enemy_picked_2, enemy_picked_3, enemy_picked_4, enemy_picked_5]

    my_picked_merge = merge_images_vertical(images=my_list)
    enemy_picked_merge = merge_images_vertical(images=enemy_list)
    total_picked_img = merge_images_horizontal(my_picked_merge, enemy_picked_merge)

    return total_picked_img


def crop_picked_champs_texts_and_portraits_area(img: Image.Image, window_size: tuple[int, int]):
    my_picked_img = crop_roi_relative_xy(img, window_size, ROI.PICKED_CHAMPIONS_MY_TEAM)
    enemy_picked_img = crop_roi_relative_xy(img, window_size, ROI.PICKED_CHAMPIONS_ENEMY_TEAM)
    total_picked_img = merge_images_horizontal(my_picked_img, enemy_picked_img)

    return total_picked_img


# ======================
# 메인 테스트 루프
# ======================
def main():
    test_case = "test_6"
    img_dir = PATHS.TEST_LOL_CLIENT_DIR / test_case
    print(img_dir)
    paths = sorted(img_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"테스트 이미지가 없음: {img_dir}")

    normalizer = TextNormalizer()
    classifier = StateClassifier()
    buffer = StateBuffer(size=7)
    state_manager = StableStateManager(min_duration=1.0, min_confidence=0.7)

    stable_state = "UNKNOWN"
    pick_coach_client = get_client()
    playplan_coach_client = get_playplan_coach_client()
    last_gemini_call_t = 0.0
    gemini_calls = 0
    did_pick_algo = False  # PICK_REAL 알고리즘 1회 실행 보장

    dual_buf = StateBuffer(size=7)

    for idx, p in enumerate(paths, start=1):
        img = Image.open(p).convert("RGB")
        window_size = (img.width, img.height)

        status_img = crop_roi_relative_xy(img, window_size, ROI.BANPICK_STATUS_TEXT)

        my_banned_img = crop_roi_relative_xy(img, window_size, ROI.BANNED_CHAMPIONS_MY_TEAM)
        enemy_banned_img = crop_roi_relative_xy(img, window_size, ROI.BANNED_CHAMPIONS_ENEMY_TEAM)

        total_picked_texts_img = crop_picked_champs_texts_area(img=img, window_size=window_size)
        total_picked_texts_and_portrait_img = crop_picked_champs_texts_and_portraits_area(
            img=img, window_size=window_size
        )

        banpick_timer_bar_img = crop_roi_relative_xy(img, window_size, ROI.BANPICK_TIMER_BAR)
        banpick_timer_digit_img = crop_roi_relative_xy(img, window_size, ROI.BANPICK_TIMER_DIGITS)

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
        print(
            f" StableState → {stable_state} | OCR={ocr!r} | norm={norm!r} | cls={cls!r} | buf={candidate}({confidence:.2f})"
        )

        if stable_state == "PICK":
            pick_res = detect_pick_kind_from_banned_strips(
                my_banned_img, enemy_banned_img, std_threshold=STD_THRESHOLD
            )
            print(" PICK 판정:", pick_res.kind, "std:", round(pick_res.std, 2))

            if pick_res.kind != "PICK_REAL":
                continue

            if did_pick_algo:
                print(" (PICK_REAL algo already executed once - skip)")
                continue

            # Gemini 호출 쿨다운 + 호출 횟수 제한
            now = time.time()
            if now - last_gemini_call_t < gemini_cooldown_sec:
                print(" (Gemini cooldown)")
                continue
            if gemini_calls >= max_gemini_calls:
                print(" (Gemini max calls reached)")
                break

            try:
                _  = run_streaming(
                    "PICK_COACH",
                    lol_mid_pick_coach_stream(
                        total_picked_texts_img, client=pick_coach_client, model="gemini-2.5-pro"
                    ),
                )

                _  = run_streaming(
                    "PICK_COACH",
                    lol_mid_pick_coach_stream(
                        total_picked_texts_and_portrait_img,
                        client=pick_coach_client,
                        model="gemini-2.5-pro",
                    ),
                )

            except Exception as e:
                print(" ❌ Gemini 호출 실패:", repr(e))
                continue

            gemini_calls += 1
            last_gemini_call_t = now

            did_pick_algo = True
            continue

        elif stable_state == "PREPARE":
            dual_now = is_dual_timer_effective(
                timer_bar_img=banpick_timer_bar_img,
                timer_digits_img=banpick_timer_digit_img,
            )

            dual_buf.push(dual_now)
            dual_stable = dual_buf.get_majority()
            dual_conf = dual_buf.get_confidence()

            print(f" DualEffective → now={dual_now} stable={dual_stable} ({dual_conf:.2f})")

            # ✅ 확정 조건: 다수결 True + 신뢰도 임계(선택)
            if dual_stable is True and dual_conf >= 0.72:
                print("양팀 모든 챔피언 픽 됐습니다 (stable)")

                _  = run_streaming(
                    "PLAYPLAN_COACH",
                    lol_playplan_stream(
                        total_picked_texts_img, client=playplan_coach_client, model="gemini-2.5-pro"
                    ),
                )

                _  = run_streaming(
                    "PLAYPLAN_COACH",
                    lol_playplan_stream(
                        total_picked_texts_and_portrait_img,
                        client=playplan_coach_client,
                        model="gemini-2.5-pro",
                    ),
                )
                break
        else:
            dual_buf.reset()

        if SLEEP_SEC:
            time.sleep(SLEEP_SEC)


if __name__ == "__main__":
    main()

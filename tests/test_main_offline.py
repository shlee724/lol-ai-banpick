# test_main_offline.py
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

from PIL import Image

from core.roi_manager import crop_roi_relative_xy
from config.roi import ROI
from config.path import PATHS

from pipeline.normalizer import TextNormalizer
from pipeline.classifier import StateClassifier
from pipeline.buffer import StateBuffer
from pipeline.state_manager import StableStateManager
from pipeline.pick_stage_detector import detect_pick_kind_from_banned_strips
from pipeline.prepare_phase_detector import is_dual_timer_effective

from core.ocr_engine import extract_text

from core.lol_pick_coach import lol_mid_pick_coach_stream, get_client
from core.lol_playplan_coach import lol_playplan_stream, get_playplan_coach_client


# ======================
# main.py와 동일한 설정
# ======================
SLEEP_SEC = 0.01
PICK_STD_THRESHOLD = 30.0

MODEL_VISION = "gemini-3-flash-preview"
MODEL_TEXT = "gemini-3-flash-preview"

MY_ROLE = "MID"
MY_TIER = "BRONZE"
MY_CHAMP_POOL = ["Malzahar", "Oriana", "Galio", "Mundo", "Garen", "Malphite", "Cho'gath", "Nasus", "kassadin"]

DEBUG_SAVE = False
pick_real_executed = False  # PICK_REAL 알고리즘 1회 실행 보장

dual_buf = StateBuffer(size=7)


def merge_images_horizontal(img1: Image.Image, img2: Image.Image, bg_color=(255, 255, 255)) -> Image.Image:
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


def _list_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    # 파일명 기준 정렬(보통 ts가 붙어있으면 시간순 됨)
    return sorted(paths, key=lambda p: p.name)


def _open_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def main() -> None:
    global pick_real_executed, dual_buf

    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", required=True, help="lol_client 하위 테스트셋 폴더명 (예: test_1)")
    parser.add_argument("--sleep", type=float, default=SLEEP_SEC, help="프레임 간 sleep (기본 0.01)")
    parser.add_argument("--limit", type=int, default=0, help="최대 처리 프레임 수 (0=무제한)")
    parser.add_argument("--no_api", action="store_true", help="제미나이 호출 없이 판정/로그만")
    args = parser.parse_args()

    test_dir = PATHS.TEST_LOL_CLIENT_DIR / args.testset
    if not test_dir.exists():
        raise FileNotFoundError(f"테스트셋 폴더 없음: {test_dir}")

    img_paths = _list_images(test_dir)
    if not img_paths:
        raise FileNotFoundError(f"이미지 없음: {test_dir}")

    # ======================
    # main.py와 동일한 객체 구성
    # ======================
    normalizer = TextNormalizer()
    classifier = StateClassifier()
    buffer = StateBuffer(size=7)
    state_manager = StableStateManager(min_duration=1.0, min_confidence=0.7)

    pick_coach_client = None
    playplan_coach_client = None
    if not args.no_api:
        pick_coach_client = get_client()
        playplan_coach_client = get_playplan_coach_client()

    pick_real_executed = False
    dual_buf = StateBuffer(size=7)

    print(f"📁 OFFLINE testset: {test_dir}")
    print(f"🖼 frames: {len(img_paths)} | no_api={args.no_api} | sleep={args.sleep}")
    print("====================================")

    processed = 0

    # ======================
    # main.py while True를 "프레임 순회"로 치환
    # ======================
    for idx, frame_path in enumerate(img_paths, start=1):
        if args.limit and processed >= args.limit:
            break

        frame_img = _open_rgb(frame_path)
        w, h = frame_img.size
        window_size = (w, h)

        # --- ROI crop (main.py 동일) ---
        status_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANPICK_STATUS_TEXT)

        if DEBUG_SAVE:
            frame_img.save(PATHS.LOL_CLIENT_CAPTURE_PNG)
            status_img.save(PATHS.BANPICK_STATUS_TEXT_CAPTURE_PNG)

        my_banned_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANNED_CHAMPIONS_MY_TEAM)
        enemy_banned_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANNED_CHAMPIONS_ENEMY_TEAM)
        my_picked_img = crop_roi_relative_xy(frame_img, window_size, ROI.PICKED_CHAMPIONS_MY_TEAM)
        enemy_picked_img = crop_roi_relative_xy(frame_img, window_size, ROI.PICKED_CHAMPIONS_ENEMY_TEAM)

        bans_merged_img = merge_images_horizontal(my_banned_img, enemy_banned_img)
        picks_merged_img = merge_images_horizontal(my_picked_img, enemy_picked_img)

        banpick_timer_bar_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANPICK_TIMER_BAR)
        banpick_timer_digit_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANPICK_TIMER_DIGITS)

        # --- OCR ---
        status_text_raw = extract_text(status_img)

        # --- Pipeline ---
        status_text_norm = normalizer.normalize(status_text_raw)
        raw_state = classifier.classify(status_text_norm)

        buffer.push(raw_state)
        major_state = buffer.get_majority()
        major_conf = buffer.get_confidence()

        stable_state = state_manager.update(major_state, major_conf)

        print(f"\n#[{idx:04d}] {frame_path.name}")
        print(f" OCR='{status_text_raw}' | norm='{status_text_norm}' | raw={raw_state} | major={major_state}({major_conf:.2f}) | stable={stable_state}")

        # ======================
        # 분기 로직(main.py 동일)
        # ======================
        if stable_state == "PICK":
            if pick_real_executed:
                processed += 1
                time.sleep(args.sleep)
                continue

            pick_res = detect_pick_kind_from_banned_strips(
                my_banned_img,
                enemy_banned_img,
                std_threshold=PICK_STD_THRESHOLD,
            )
            print("PICK 판정:", pick_res.kind, "std:", round(pick_res.std, 2))

            if pick_res.kind == "PICK_REAL":
                if args.no_api:
                    print(" (no_api) PICK_REAL 감지 - 스트리밍 호출 생략")
                    pick_real_executed = True
                    processed += 1
                    time.sleep(args.sleep)
                    continue

                try:
                    _final_text = run_streaming(
                        "PICK_COACH",
                        lol_mid_pick_coach_stream(
                            picks_merged_img,
                            client=pick_coach_client,
                            model="gemini-2.5-pro",
                        ),
                    )
                except Exception as e:
                    print(" ❌ Gemini 호출 실패:", repr(e))
                    processed += 1
                    time.sleep(args.sleep)
                    continue

                pick_real_executed = True
                processed += 1
                time.sleep(args.sleep)
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

            if dual_stable is True and dual_conf >= 0.72:
                print("양팀 모든 챔피언 픽 됐습니다 (stable)")

                if args.no_api:
                    print(" (no_api) PLAYPLAN 스트리밍 호출 생략 - 종료")
                    break

                _final_text = run_streaming(
                    "PLAYPLAN_COACH",
                    lol_playplan_stream(
                        picks_merged_img,
                        client=playplan_coach_client,
                        model="gemini-2.5-pro",
                    ),
                )
                break

        else:
            # main.py: PREPARE가 아니면 dual_buf.reset()
            dual_buf.reset()

        processed += 1
        time.sleep(args.sleep)

    print("\n====================================")
    print(f"✅ OFFLINE DONE. processed={processed} / total_frames={len(img_paths)}")


if __name__ == "__main__":
    main()
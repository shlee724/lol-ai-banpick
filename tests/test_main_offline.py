# test_main_offline.py
from __future__ import annotations

# ======================
# Standard library
# ======================
import argparse
import time
from pathlib import Path
from typing import Iterable, Optional

# ======================
# Third-party
# ======================
from PIL import Image

# ======================
# Local modules
# ======================
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
# Config / Constants (main.py와 동일 톤)
# ======================
SLEEP_SEC = 0.01

STATE_BUF_SIZE = 7
DUAL_BUF_SIZE = 7

PICK_STD_THRESHOLD = 30.0
DUAL_CONF_THRESHOLD = 0.72

GEMINI_MODEL = "gemini-2.5-pro"

MY_ROLE = "MID"
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

DEBUG_SAVE = False


# ======================
# Helpers
# ======================
def merge_images_horizontal(img_left: Image.Image, img_right: Image.Image, bg_color=(255, 255, 255)) -> Image.Image:
    """Create a new image by placing img_left and img_right side-by-side."""
    new_width = img_left.width + img_right.width
    new_height = max(img_left.height, img_right.height)

    merged = Image.new("RGB", (new_width, new_height), bg_color)
    merged.paste(img_left, (0, 0))
    merged.paste(img_right, (img_left.width, 0))
    return merged


def run_streaming(label: str, stream_iter: Iterable[str]) -> str:
    """Consume streaming text deltas, print them, and return full concatenated text."""
    chunks: list[str] = []

    start_t = time.perf_counter()
    first_token_time: Optional[float] = None

    for delta in stream_iter:
        if first_token_time is None:
            first_token_time = time.perf_counter()
            print(f"\n[{label}] ⏱ 첫 토큰: {first_token_time - start_t:.2f}s\n")
        print(delta, end="", flush=True)
        chunks.append(delta)

    end_t = time.perf_counter()
    print(f"\n\n[{label}] ⏱ 전체: {end_t - start_t:.2f}s")
    return "".join(chunks)


def list_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths, key=lambda p: p.name)


def open_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ======================
# Main
# ======================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", required=True, help="lol_client 하위 테스트셋 폴더명 (예: test_1)")
    parser.add_argument("--sleep", type=float, default=SLEEP_SEC, help="프레임 간 sleep (기본 0.01)")
    parser.add_argument("--limit", type=int, default=0, help="최대 처리 프레임 수 (0=무제한)")
    parser.add_argument("--no_api", action="store_true", help="제미나이 호출 없이 판정/로그만")
    args = parser.parse_args()

    test_dir = PATHS.TEST_LOL_CLIENT_DIR / args.testset
    if not test_dir.exists():
        raise FileNotFoundError(f"테스트셋 폴더 없음: {test_dir}")

    img_paths = list_images(test_dir)
    if not img_paths:
        raise FileNotFoundError(f"이미지 없음: {test_dir}")

    # ----------------------
    # Init (main.py와 동일 톤)
    # ----------------------
    normalizer = TextNormalizer()
    classifier = StateClassifier()

    state_buf = StateBuffer(size=STATE_BUF_SIZE)
    dual_buf = StateBuffer(size=DUAL_BUF_SIZE)

    state_manager = StableStateManager(min_duration=1.0, min_confidence=0.7)

    pick_coach_client = None
    playplan_coach_client = None
    if not args.no_api:
        pick_coach_client = get_client()
        playplan_coach_client = get_playplan_coach_client()

    pick_real_executed = False  # PICK_REAL 알고리즘 1회 실행 보장

    print(f"📁 OFFLINE testset: {test_dir}")
    print(f"🖼 frames: {len(img_paths)} | no_api={args.no_api} | sleep={args.sleep}")
    print("====================================")

    processed = 0

    # ----------------------
    # main.py while True를 "프레임 순회"로 치환
    # ----------------------
    for idx, frame_path in enumerate(img_paths, start=1):
        if args.limit and processed >= args.limit:
            break

        # ----------------------
        # 1) Frame load
        # ----------------------
        frame_img = open_rgb(frame_path)
        w, h = frame_img.size
        window_size = (w, h)

        # ----------------------
        # 2) ROI extraction
        # ----------------------
        status_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANPICK_STATUS_TEXT)

        bans_my_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANNED_CHAMPIONS_MY_TEAM)
        bans_enemy_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANNED_CHAMPIONS_ENEMY_TEAM)

        picks_my_img = crop_roi_relative_xy(frame_img, window_size, ROI.PICKED_CHAMPIONS_MY_TEAM)
        picks_enemy_img = crop_roi_relative_xy(frame_img, window_size, ROI.PICKED_CHAMPIONS_ENEMY_TEAM)

        bans_merged_img = merge_images_horizontal(bans_my_img, bans_enemy_img)
        picks_merged_img = merge_images_horizontal(picks_my_img, picks_enemy_img)

        timer_bar_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANPICK_TIMER_BAR)
        timer_digits_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANPICK_TIMER_DIGITS)

        if DEBUG_SAVE:
            frame_img.save(PATHS.LOL_CLIENT_CAPTURE_PNG)
            status_img.save(PATHS.BANPICK_STATUS_TEXT_CAPTURE_PNG)

        # ----------------------
        # 3) OCR + state pipeline
        # ----------------------
        status_text_raw = extract_text(status_img)
        status_text_norm = normalizer.normalize(status_text_raw)
        raw_state = classifier.classify(status_text_norm)

        state_buf.push(raw_state)
        major_state = state_buf.get_majority()
        major_conf = state_buf.get_confidence()

        stable_state = state_manager.update(major_state, major_conf)

        print(f"\n#[{idx:04d}] {frame_path.name}")
        print(
            f" OCR='{status_text_raw}' | norm='{status_text_norm}'"
            f" | raw={raw_state} | major={major_state}({major_conf:.2f}) | stable={stable_state}"
        )

        # ----------------------
        # 4) State actions (main.py와 동일)
        # ----------------------
        if stable_state == "PICK":
            # Guards
            if raw_state == "BAN":
                processed += 1
                time.sleep(args.sleep)
                continue

            if pick_real_executed:
                processed += 1
                time.sleep(args.sleep)
                continue

            # Detect pick stage
            pick_res = detect_pick_kind_from_banned_strips(
                bans_my_img,
                bans_enemy_img,
                std_threshold=PICK_STD_THRESHOLD,
            )
            print(f"[PICK] 판정: kind={pick_res.kind} std={pick_res.std:.2f}")

            # Call pick coach
            if pick_res.kind == "PICK_REAL":
                if args.no_api:
                    print("[PICK] (no_api) PICK_REAL 감지 - 스트리밍 호출 생략")
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
                            model=GEMINI_MODEL,
                        ),
                    )
                except Exception as e:
                    print("[ERR] Gemini 호출 실패:", repr(e))
                    processed += 1
                    time.sleep(args.sleep)
                    continue

                pick_real_executed = True
                processed += 1
                time.sleep(args.sleep)
                continue

        elif stable_state == "PREPARE":
            dual_now = is_dual_timer_effective(
                timer_bar_img=timer_bar_img,
                timer_digits_img=timer_digits_img,
            )

            dual_buf.push(dual_now)
            dual_stable = dual_buf.get_majority()
            dual_conf = dual_buf.get_confidence()

            print(f"[PREPARE] DualEffective: now={dual_now} stable={dual_stable} ({dual_conf:.2f})")

            if dual_stable is True and dual_conf >= DUAL_CONF_THRESHOLD:
                print("[PREPARE] 양팀 모든 챔피언 픽 됐습니다 (stable)")

                if args.no_api:
                    print("[PREPARE] (no_api) PLAYPLAN 스트리밍 호출 생략 - 종료")
                    break

                _final_text = run_streaming(
                    "PLAYPLAN_COACH",
                    lol_playplan_stream(
                        picks_merged_img,
                        client=playplan_coach_client,
                        model=GEMINI_MODEL,
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
from __future__ import annotations

# ======================
# Standard library
# ======================
import time
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
from core.screen_capture import capture_window
from core.window_tracker import WindowTracker
from pipeline.buffer import StateBuffer
from pipeline.classifier import StateClassifier
from pipeline.normalizer import TextNormalizer
from pipeline.pick_stage_detector import detect_pick_kind_from_banned_strips
from pipeline.prepare_phase_detector import is_dual_timer_effective
from pipeline.state_manager import StableStateManager

# ======================
# Config / Constants
# ======================
SLEEP_SEC = 0.01

STATE_BUF_SIZE = 7
DUAL_BUF_SIZE = 7

PICK_STD_THRESHOLD = 30.0
DUAL_CONF_THRESHOLD = 0.72

GEMINI_MODEL = "gemini-2.5-pro"

DEBUG_SAVE = False

WINDOW_TITLE = "League of Legends"


# ======================
# Helpers
# ======================
def merge_images_horizontal(
    img_left: Image.Image, img_right: Image.Image, bg_color=(255, 255, 255)
) -> Image.Image:
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


# ======================
# Init (global objects)
# ======================
tracker = WindowTracker(WINDOW_TITLE)

normalizer = TextNormalizer()
classifier = StateClassifier()

state_buf = StateBuffer(size=STATE_BUF_SIZE)
dual_buf = StateBuffer(size=DUAL_BUF_SIZE)

state_manager = StableStateManager(min_duration=1.0, min_confidence=0.7)

pick_coach_client = get_client()
playplan_coach_client = get_playplan_coach_client()

pick_real_executed = False  # PICK_REAL 알고리즘 1회 실행 보장


# ======================
# Main loop
# ======================
while True:
    # ----------------------
    # 1) Window / frame capture
    # ----------------------
    window_rect = tracker.get_window_rect()
    if window_rect is None or not tracker.hwnd:
        print("[WARN] 롤 클라이언트를 찾을 수 없음")
        dual_buf.reset()
        time.sleep(SLEEP_SEC)
        continue

    x, y, w, h = window_rect
    window_size = (w, h)

    frame_img = capture_window(tracker.hwnd, w, h)
    if frame_img is None:
        print("[WARN] 화면 캡처 실패")
        dual_buf.reset()
        time.sleep(SLEEP_SEC)
        continue

    # ----------------------
    # 2) ROI extraction
    # ----------------------
    status_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANPICK_STATUS_TEXT)

    bans_my_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANNED_CHAMPIONS_MY_TEAM)
    bans_enemy_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANNED_CHAMPIONS_ENEMY_TEAM)

    picks_my_img = crop_roi_relative_xy(frame_img, window_size, ROI.PICKED_CHAMPIONS_MY_TEAM)
    picks_enemy_img = crop_roi_relative_xy(frame_img, window_size, ROI.PICKED_CHAMPIONS_ENEMY_TEAM)

    #bans_merged_img = merge_images_horizontal(bans_my_img, bans_enemy_img)
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

    # ----------------------
    # 4) State actions
    # ----------------------
    if stable_state == "PICK":
        # Guards
        if raw_state == "BAN":
            time.sleep(SLEEP_SEC)
            continue

        if pick_real_executed:
            time.sleep(SLEEP_SEC)
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
                time.sleep(SLEEP_SEC)
                continue

            pick_real_executed = True
            time.sleep(SLEEP_SEC)
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
        dual_buf.reset()

    # ----------------------
    # 5) Sleep
    # ----------------------
    time.sleep(SLEEP_SEC)

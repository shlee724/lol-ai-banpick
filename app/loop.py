from __future__ import annotations
import time
from typing import Iterable, Optional

from config.path import PATHS
from core.ocr_engine import extract_text
from core.lol_pick_coach import get_client, lol_mid_pick_coach_stream
from core.lol_playplan_coach import get_playplan_coach_client, lol_playplan_stream

from pipeline.buffer import StateBuffer
from pipeline.classifier import StateClassifier
from pipeline.normalizer import TextNormalizer
from pipeline.pick_stage_detector import detect_pick_kind_from_banned_strips
from pipeline.prepare_phase_detector import is_dual_timer_effective
from pipeline.state_manager import StableStateManager

from core.window_tracker import WindowTracker

from app.settings import Settings
from app.capture import get_frame
from app.rois import extract_rois

def run_streaming(label: str, stream_iter: Iterable[str]) -> str:
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

def run_main(settings: Settings) -> None:
    tracker = WindowTracker(settings.window_title)

    normalizer = TextNormalizer()
    classifier = StateClassifier()

    state_buf = StateBuffer(size=settings.state_buf_size)
    dual_buf = StateBuffer(size=settings.dual_buf_size)

    state_manager = StableStateManager(min_duration=1.0, min_confidence=0.7)

    pick_coach_client = get_client()
    playplan_coach_client = get_playplan_coach_client()

    pick_real_executed = False

    while True:
        frame_img, window_size = get_frame(tracker, settings.sleep_sec)
        if frame_img is None or window_size is None:
            print("[WARN] 롤 클라이언트를 찾을 수 없음/캡처 실패")
            dual_buf.reset()
            time.sleep(settings.sleep_sec)
            continue

        rois = extract_rois(frame_img, window_size)

        if settings.debug_save:
            frame_img.save(PATHS.LOL_CLIENT_CAPTURE_PNG)
            rois.status_img.save(PATHS.BANPICK_STATUS_TEXT_CAPTURE_PNG)

        status_text_raw = extract_text(rois.status_img)
        status_text_norm = normalizer.normalize(status_text_raw)
        raw_state = classifier.classify(status_text_norm)

        state_buf.push(raw_state)
        major_state = state_buf.get_majority()
        major_conf = state_buf.get_confidence()
        stable_state = state_manager.update(major_state, major_conf)
        print("stable state:", stable_state)

        if stable_state == "PICK":
            if raw_state == "BAN" or pick_real_executed:
                time.sleep(settings.sleep_sec)
                continue

            pick_res = detect_pick_kind_from_banned_strips(
                rois.bans_my_img,
                rois.bans_enemy_img,
                std_threshold=settings.pick_std_threshold,
            )
            print(f"[PICK] 판정: kind={pick_res.kind} std={pick_res.std:.2f}")

            if pick_res.kind == "PICK_REAL":
                try:
                    run_streaming(
                        "PICK_COACH",
                        lol_mid_pick_coach_stream(
                            rois.picks_merged_img,
                            client=pick_coach_client,
                            model=settings.gemini_model,
                        ),
                    )
                except Exception as e:
                    print("[ERR] Gemini 호출 실패:", repr(e))
                    time.sleep(settings.sleep_sec)
                    continue

                pick_real_executed = True
                time.sleep(settings.sleep_sec)
                continue

        elif stable_state == "PREPARE":
            dual_now = is_dual_timer_effective(
                timer_bar_img=rois.timer_bar_img,
                timer_digits_img=rois.timer_digits_img,
            )
            dual_buf.push(dual_now)
            dual_stable = dual_buf.get_majority()
            dual_conf = dual_buf.get_confidence()

            print(f"[PREPARE] DualEffective: now={dual_now} stable={dual_stable} ({dual_conf:.2f})")

            if dual_stable is True and dual_conf >= settings.dual_conf_threshold:
                print("[PREPARE] 양팀 모든 챔피언 픽 됐습니다 (stable)")
                run_streaming(
                    "PLAYPLAN_COACH",
                    lol_playplan_stream(
                        rois.picks_merged_img,
                        client=playplan_coach_client,
                        model=settings.gemini_model,
                    ),
                )
                break
        else:
            dual_buf.reset()

        time.sleep(settings.sleep_sec)
from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

from PIL import Image
from core.screen_capture import capture_window
from config.path import PATHS
from app.settings import AppSettings
from app.wiring import AppDeps
from app.frame_types import FrameContext


def run_loop(
    settings: AppSettings,
    deps: AppDeps,
    stop_event: Optional[object] = None,  # threading.Event 같은 것(UI 대비)
) -> None:
    while True:
        if stop_event is not None and getattr(stop_event, "is_set")():
            print("[INFO] stop_event set -> exit loop")
            return

        # 1) Window / frame capture
        window_rect = deps.tracker.get_window_rect()
        if window_rect is None or not deps.tracker.hwnd:
            print("[WARN] 롤 클라이언트를 찾을 수 없음")
            deps.dual_buf.reset()
            time.sleep(settings.sleep_sec)
            continue

        _, _, w, h = window_rect
        window_size = (w, h)

        frame_img = capture_window(deps.tracker.hwnd, w, h)
        if frame_img is None:
            print("[WARN] 화면 캡처 실패")
            deps.dual_buf.reset()
            time.sleep(settings.sleep_sec)
            continue

        # 2) ROI extraction
        rois = deps.roi_extractor.extract(frame_img, window_size)

        if settings.debug_save:
            frame_img.save(PATHS.LOL_CLIENT_CAPTURE_PNG)
            rois.status_img.save(PATHS.BANPICK_STATUS_TEXT_CAPTURE_PNG)

        # 3) OCR + state pipeline
        state = deps.state_estimator.estimate(rois.status_img)

        ctx = FrameContext(rois=rois, state=state)

        # 4) Workflow (현재: solo)
        if state.stable_state == "PICK":
            res = deps.workflow.on_pick(ctx)
            if res.should_stop:
                return

        elif state.stable_state == "PREPARE":
            dual_now = deps.workflow.get_dual_now(ctx)

            deps.dual_buf.push(dual_now)
            dual_stable = deps.dual_buf.get_majority()
            dual_conf = deps.dual_buf.get_confidence()

            print(f"[PREPARE] DualEffective: now={dual_now} stable={dual_stable} ({dual_conf:.2f})")

            if dual_stable is True and dual_conf >= settings.dual_conf_threshold:
                res = deps.workflow.on_prepare_dual_stable(ctx)
                if res.should_stop:
                    return

        else:
            deps.dual_buf.reset()

        time.sleep(settings.sleep_sec)


FrameProvider = Callable[[], Optional[Tuple[Image.Image, Tuple[int, int]]]]
# 반환: (frame_img, (w,h)) / 더 이상 프레임 없으면 None

def run_loop_with_provider(
    settings: AppSettings,
    deps: AppDeps,
    frame_provider: FrameProvider,
    sleep_sec: Optional[float] = None,
    stop_event: Optional[object] = None,  # UI 대비
) -> None:
    sleep = settings.sleep_sec if sleep_sec is None else sleep_sec

    while True:
        if stop_event is not None and getattr(stop_event, "is_set")():
            print("[INFO] stop_event set -> exit loop")
            return

        provided = frame_provider()
        if provided is None:
            print("[INFO] no more frames -> exit loop")
            return

        frame_img, window_size = provided

        rois = deps.roi_extractor.extract(frame_img, window_size)

        if settings.debug_save:
            frame_img.save(PATHS.LOL_CLIENT_CAPTURE_PNG)
            rois.status_img.save(PATHS.BANPICK_STATUS_TEXT_CAPTURE_PNG)

        state = deps.state_estimator.estimate(rois.status_img)

        ctx = FrameContext(rois=rois, state=state)

        if state.stable_state == "PICK":
            res = deps.workflow.on_pick(ctx)
            if res.should_stop:
                return

        elif state.stable_state == "PREPARE":
            dual_now = deps.workflow.get_dual_now(ctx)

            deps.dual_buf.push(dual_now)
            dual_stable = deps.dual_buf.get_majority()
            dual_conf = deps.dual_buf.get_confidence()

            print(f"[PREPARE] DualEffective: now={dual_now} stable={dual_stable} ({dual_conf:.2f})")

            if dual_stable is True and dual_conf >= settings.dual_conf_threshold:
                res = deps.workflow.on_prepare_dual_stable(ctx)
                if res.should_stop:
                    return

        else:
            deps.dual_buf.reset()

        time.sleep(sleep)
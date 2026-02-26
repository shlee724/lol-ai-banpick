
from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

from PIL import Image
from core.screen_capture import capture_window
from config.path import PATHS
from app.settings import AppSettings
from app.wiring import AppDeps
from app.frame_types import FrameContext
from app.command_runner import CommandRunner

def run_loop(settings: AppSettings, deps: AppDeps, stop_event: Optional[object] = None) -> None:
    runner = CommandRunner(settings=settings, deps=deps)

    while True:
        if stop_event is not None and getattr(stop_event, "is_set")():
            print("[INFO] stop_event set -> exit loop")
            return

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

        rois = deps.roi_extractor.extract(frame_img, window_size)

        if settings.debug_save:
            frame_img.save(PATHS.LOL_CLIENT_CAPTURE_PNG)
            rois.status_img.save(PATHS.BANPICK_STATUS_TEXT_CAPTURE_PNG)

        state = deps.state_estimator.estimate(rois.status_img)
        ctx = FrameContext(rois=rois, state=state)

        commands = deps.workflow.on_frame(ctx)
        should_stop = runner.execute(ctx, commands)
        if should_stop:
            return

        time.sleep(settings.sleep_sec)

FrameProvider = Callable[[], Optional[Tuple[Image.Image, Tuple[int, int], str]]]


def run_loop_with_provider(
    settings: AppSettings,
    deps: AppDeps,
    frame_provider: FrameProvider,
    *,
    sleep_sec: Optional[float] = None,
    stop_event: Optional[object] = None,
) -> None:
    """
    오프라인/재생용 루프.
    - run_loop와 동일 엔진(ROI->State->Workflow->Commands->Runner)을 사용
    - 캡처 대신 frame_provider가 프레임을 공급
    - 더 이상 프레임이 없으면 종료
    """
    runner = CommandRunner(settings=settings, deps=deps)
    sleep = settings.sleep_sec if sleep_sec is None else sleep_sec

    while True:
        if stop_event is not None and getattr(stop_event, "is_set")():
            print("[INFO] stop_event set -> exit loop")
            return

        provided = frame_provider()
        if provided is None:
            print("[INFO] no more frames -> exit loop")
            return

        frame_img, window_size, frame_id = provided

        # (선택) 프레임 식별 로그가 필요하면 여기서 찍으면 됨
        # print(f"\n[FRAME] {frame_id}")

        rois = deps.roi_extractor.extract(frame_img, window_size)

        if settings.debug_save:
            frame_img.save(PATHS.LOL_CLIENT_CAPTURE_PNG)
            rois.status_img.save(PATHS.BANPICK_STATUS_TEXT_CAPTURE_PNG)

        state = deps.state_estimator.estimate(rois.status_img)
        ctx = FrameContext(rois=rois, state=state)

        commands = deps.workflow.on_frame(ctx)
        should_stop = runner.execute(ctx, commands)
        if should_stop:
            return

        time.sleep(sleep)
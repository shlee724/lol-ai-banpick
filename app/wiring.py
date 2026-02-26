from __future__ import annotations

from dataclasses import dataclass

from app.settings import AppSettings
from app.roi_extractor import FrameRoiExtractor
from app.state_estimator import StateEstimator

from core.window_tracker import WindowTracker
from pipeline.normalizer import TextNormalizer
from pipeline.classifier import StateClassifier
from pipeline.buffer import StateBuffer
from pipeline.state_manager import StableStateManager

from core.lol_pick_coach import get_client
from core.lol_playplan_coach import get_playplan_coach_client

from workflow.solo import SoloWorkflow


@dataclass
class AppDeps:
    tracker: WindowTracker
    roi_extractor: FrameRoiExtractor
    state_estimator: StateEstimator
    dual_buf: StateBuffer

    pick_client: object
    playplan_client: object

    workflow: SoloWorkflow  # 지금은 solo만. 추후 UI 선택에 따라 다른 workflow를 꽂으면 됨.


def build_deps(settings: AppSettings) -> AppDeps:
    tracker = WindowTracker(settings.window_title)

    normalizer = TextNormalizer()
    classifier = StateClassifier()
    state_buf = StateBuffer(size=settings.state_buf_size)
    state_manager = StableStateManager(min_duration=1.0, min_confidence=0.7)

    roi_extractor = FrameRoiExtractor()
    state_estimator = StateEstimator(
        normalizer=normalizer,
        classifier=classifier,
        state_buf=state_buf,
        state_manager=state_manager,
    )

    dual_buf = StateBuffer(size=settings.dual_buf_size)

    pick_client = get_client()
    playplan_client = get_playplan_coach_client()

    workflow = SoloWorkflow(
        pick_std_threshold=settings.pick_std_threshold,
        dual_conf_threshold=settings.dual_conf_threshold,
    )

    return AppDeps(
        pick_client=pick_client,
        playplan_client=playplan_client,
        tracker=tracker,
        roi_extractor=roi_extractor,
        state_estimator=state_estimator,
        dual_buf=dual_buf,
        workflow=workflow,
    )
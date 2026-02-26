from __future__ import annotations

from dataclasses import dataclass
from PIL import Image

from core.ocr_engine import extract_text
from pipeline.buffer import StateBuffer
from pipeline.classifier import StateClassifier
from pipeline.normalizer import TextNormalizer
from pipeline.state_manager import StableStateManager

from app.frame_types import StateSnapshot


@dataclass
class StateEstimator:
    normalizer: TextNormalizer
    classifier: StateClassifier
    state_buf: StateBuffer
    state_manager: StableStateManager

    def estimate(self, status_img: Image.Image) -> StateSnapshot:
        status_text_raw = extract_text(status_img)
        status_text_norm = self.normalizer.normalize(status_text_raw)
        raw_state = self.classifier.classify(status_text_norm)

        self.state_buf.push(raw_state)
        major_state = self.state_buf.get_majority()
        major_conf = self.state_buf.get_confidence()

        stable_state = self.state_manager.update(major_state, major_conf)

        return StateSnapshot(
            raw_state=raw_state,
            major_state=major_state,
            major_conf=major_conf,
            stable_state=stable_state,
        )
from __future__ import annotations
from dataclasses import dataclass
from PIL import Image

@dataclass(frozen=True)
class FrameRois:
    status_img: Image.Image
    bans_my_img: Image.Image
    bans_enemy_img: Image.Image
    picks_my_img: Image.Image
    picks_enemy_img: Image.Image
    picks_merged_img: Image.Image
    timer_bar_img: Image.Image
    timer_digits_img: Image.Image

@dataclass(frozen=True)
class StateSnapshot:
    raw_state: str
    major_state: str
    major_conf: float
    stable_state: str

@dataclass(frozen=True)
class FrameContext:
    rois: FrameRois
    state: StateSnapshot
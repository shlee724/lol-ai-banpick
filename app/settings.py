from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    sleep_sec: float = 0.01

    state_buf_size: int = 7
    dual_buf_size: int = 7

    pick_std_threshold: float = 30.0
    dual_conf_threshold: float = 0.72

    gemini_model: str = "gemini-2.5-pro"
    debug_save: bool = False

    window_title: str = "League of Legends"
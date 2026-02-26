# pipeline/ban_detector.py
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class BanStripDetectResult:
    std: float
    is_filled: bool


def detect_ban_strip_variance(
    ban_strip_roi: Image.Image,
    *,
    std_threshold: float = 30.0,
) -> BanStripDetectResult:
    """
    ban_strip_roi: banned_champions_area_my_team / enemy_team ROI 이미지(PIL)

    std_threshold:
      - 높일수록 보수적(확실히 초상화가 보일 때만 filled)
      - 낮출수록 민감(빈 슬롯도 filled로 오탐 가능)
    """
    # RGB -> GRAY
    rgb = np.array(ban_strip_roi.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 너무 작은 잡음/압축 노이즈 완화
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 테두리/프레임 UI가 std를 올릴 수 있으니 내부만 살짝 크롭
    h, w = gray.shape[:2]
    pad_x = int(w * 0.08)
    pad_y = int(h * 0.15)
    inner = (
        gray[pad_y : h - pad_y, pad_x : w - pad_x]
        if (h - 2 * pad_y > 5 and w - 2 * pad_x > 5)
        else gray
    )

    s = float(np.std(inner))
    is_filled = s >= std_threshold

    return BanStripDetectResult(std=s, is_filled=is_filled)

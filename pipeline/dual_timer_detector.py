from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import cv2
from PIL import Image


# ======================
# Config
# ======================
@dataclass(frozen=True)
class SymmetryConfig:
    center_ignore_ratio: float = 0.18
    smooth_window: int = 9

    # HSV 채도 균형
    sat_balance_threshold: float = 0.82

    # 그레이 대칭
    ncc_threshold: float = 0.55
    l1_threshold: float = 0.80
    min_profile_std: float = 1.0

    # ✅ NEW: Lab chroma(색 진함) 균형 필터
    # single(오른쪽만 빨강)처럼 한쪽만 색이 강하면 크게 떨어짐
    chroma_balance_threshold: float = 0.70

    # (옵션) 양쪽 모두 “강한 chroma 컬럼”이 조금이라도 있어야 dual 후보로 인정
    # 너무 높게 잡으면 dual에서도 떨어질 수 있어 낮게 시작
    chroma_presence_threshold: float = 0.02
    chroma_presence_quantile: float = 0.85


# ======================
# Utils
# ======================
def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _moving_average_1d(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.astype(np.float32)
    win = _clamp_int(int(win), 1, 101)
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x.astype(np.float32), k, mode="same").astype(np.float32)


def _zscore_1d(x: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, float]:
    x = x.astype(np.float32)
    m = float(x.mean())
    s = float(x.std())
    return (x - m) / (s + eps), s


def _split_left_right_1d(x: np.ndarray, center_ignore_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    w = int(x.shape[0])
    ci = float(center_ignore_ratio)
    ci = max(0.0, min(ci, 0.6))

    l_end = int(w * (0.5 - ci / 2.0))
    r_start = int(w * (0.5 + ci / 2.0))
    l_end = _clamp_int(l_end, 1, w - 1)
    r_start = _clamp_int(r_start, 1, w - 1)

    return x[:l_end], x[r_start:]


def _gray_symmetry_scores(gray: np.ndarray, cfg: SymmetryConfig) -> Tuple[float, float]:
    prof = gray.mean(axis=0)
    prof = _moving_average_1d(prof, cfg.smooth_window)

    left, right = _split_left_right_1d(prof, cfg.center_ignore_ratio)

    m = int(min(left.shape[0], right.shape[0]))
    if m < 10:
        return -1.0, 999.0

    left = left[:m]
    right = right[-m:][::-1]

    left_z, left_std = _zscore_1d(left)
    right_z, right_std = _zscore_1d(right)

    if left_std < cfg.min_profile_std or right_std < cfg.min_profile_std:
        return -1.0, 999.0

    ncc = float(np.mean(left_z * right_z))
    l1 = float(np.mean(np.abs(left_z - right_z)))
    return ncc, l1


def _sat_balance(hsv: np.ndarray, cfg: SymmetryConfig) -> float:
    s = hsv[:, :, 1].astype(np.float32)
    s_prof = s.mean(axis=0)

    left, right = _split_left_right_1d(s_prof, cfg.center_ignore_ratio)

    l_sum = float(np.sum(left))
    r_sum = float(np.sum(right))

    return float(min(l_sum, r_sum) / (max(l_sum, r_sum) + 1e-9))


def _lab_chroma_balance(rgb_arr: np.ndarray, cfg: SymmetryConfig) -> Tuple[float, float]:
    """
    ✅ 핵심 추가:
    Lab의 chroma(C = sqrt(a^2 + b^2))로 좌/우 색 진함 균형을 본다.
    - 배경이 하늘색으로 대칭이어도, 한쪽만 빨강 바가 강하면 imbalance가 확 떨어짐.
    """
    lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB).astype(np.float32)

    # OpenCV Lab: L(0~255), a(0~255), b(0~255) with 128 offset
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    chroma = np.sqrt(a * a + b * b)  # (H, W)

    chroma_prof = chroma.mean(axis=0).astype(np.float32)  # (W,)

    left, right = _split_left_right_1d(chroma_prof, cfg.center_ignore_ratio)

    l_sum = float(np.sum(left))
    r_sum = float(np.sum(right))
    bal = float(min(l_sum, r_sum) / (max(l_sum, r_sum) + 1e-9))

    # presence: "chroma가 큰 컬럼"이 좌/우 모두 조금이라도 존재하는지
    thr = float(np.quantile(chroma_prof, cfg.chroma_presence_quantile))
    l_pres = float(np.mean(left > thr))
    r_pres = float(np.mean(right > thr))
    pres = float(min(l_pres, r_pres))

    return bal, pres


# ======================
# Public API
# ======================
def is_dual_sided_timer_cropped_symmetry(img: Image.Image, cfg: SymmetryConfig = SymmetryConfig()) -> bool:
    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8)

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # ✅ 0) Lab chroma로 “한쪽만 강한 색”인 single을 먼저 컷
    chroma_bal, chroma_pres = _lab_chroma_balance(arr, cfg)
    if chroma_bal < cfg.chroma_balance_threshold:
        return False
    if chroma_pres < cfg.chroma_presence_threshold:
        # 양쪽 모두 '색 진함이 큰 컬럼'이 거의 없으면,
        # 단색 배경 영향으로 대칭 점수만 좋아지는 오탐을 막는 안전장치
        return False

    # 1) 채도 균형
    sat_bal = _sat_balance(hsv, cfg)
    if sat_bal < cfg.sat_balance_threshold:
        return False

    # 2) 명도 대칭
    ncc, l1 = _gray_symmetry_scores(gray, cfg)
    return (ncc >= cfg.ncc_threshold) and (l1 <= cfg.l1_threshold)


def is_dual_sided_timer_cropped(img: Image.Image) -> bool:
    return is_dual_sided_timer_cropped_symmetry(img)
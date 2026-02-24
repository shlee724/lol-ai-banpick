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
    # 중앙 숫자/세로선 영향 제거: 가운데 일정 비율은 버리고 비교
    center_ignore_ratio: float = 0.18

    # 1D 프로파일 스무딩(노이즈 완화)
    smooth_window: int = 9

    # ---- (핵심) 채도 균형 임계값 ----
    # sat_balance = min(sum(S_left), sum(S_right)) / max(...)
    # dual은 대체로 0.85~1.0, single은 0.4~0.7 근처로 떨어지는 경향
    sat_balance_threshold: float = 0.82

    # ---- 명도(그레이) 대칭 임계값 ----
    # 너무 빡세게 잡으면 prepare(어두운 단계)에서 놓칠 수 있어서 완화
    ncc_threshold: float = 0.55   # 높을수록 대칭
    l1_threshold: float = 0.80    # 낮을수록 대칭

    # 너무 평평한(정보가 거의 없는) 경우 안전장치
    min_profile_std: float = 1.0


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
    """
    x: (W,) 1D signal
    return: left, right (right is NOT flipped yet)
    """
    w = int(x.shape[0])
    ci = float(center_ignore_ratio)
    ci = max(0.0, min(ci, 0.6))

    l_end = int(w * (0.5 - ci / 2.0))
    r_start = int(w * (0.5 + ci / 2.0))
    l_end = _clamp_int(l_end, 1, w - 1)
    r_start = _clamp_int(r_start, 1, w - 1)

    left = x[:l_end]
    right = x[r_start:]
    return left, right


def _gray_symmetry_scores(gray: np.ndarray, cfg: SymmetryConfig) -> Tuple[float, float]:
    """
    gray: (H, W) float32
    - 세로 평균 -> 1D 명도 프로파일
    - 우측 flip 후 좌측과 비교
    - NCC, L1 산출
    """
    prof = gray.mean(axis=0)  # (W,)
    prof = _moving_average_1d(prof, cfg.smooth_window)

    left, right = _split_left_right_1d(prof, cfg.center_ignore_ratio)

    m = int(min(left.shape[0], right.shape[0]))
    if m < 10:
        return -1.0, 999.0

    left = left[:m]
    right = right[-m:][::-1]  # right flip

    left_z, left_std = _zscore_1d(left)
    right_z, right_std = _zscore_1d(right)

    if left_std < cfg.min_profile_std or right_std < cfg.min_profile_std:
        return -1.0, 999.0

    ncc = float(np.mean(left_z * right_z))                # [-1, 1]
    l1 = float(np.mean(np.abs(left_z - right_z)))         # [0, ...]
    return ncc, l1


def _sat_balance(hsv: np.ndarray, cfg: SymmetryConfig) -> float:
    """
    hsv: (H, W, 3) float32/uint8
    sat_balance = min(sum(S_left), sum(S_right)) / max(...)
    """
    s = hsv[:, :, 1].astype(np.float32)
    s_prof = s.mean(axis=0)  # (W,)

    left, right = _split_left_right_1d(s_prof, cfg.center_ignore_ratio)

    # 총 채도(면적) 비교
    l_sum = float(np.sum(left))
    r_sum = float(np.sum(right))

    denom = max(l_sum, r_sum) + 1e-9
    return float(min(l_sum, r_sum) / denom)


# ======================
# Public API
# ======================
def is_dual_sided_timer_cropped_symmetry(img: Image.Image, cfg: SymmetryConfig = SymmetryConfig()) -> bool:
    """
    ROI가 거의 '바만' 크롭된 이미지에서 듀얼 타이머(양쪽 바) 판정.
    핵심: (채도 균형이 높음) AND (명도 대칭이 충분함)
    """
    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8)

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)

    sat_bal = _sat_balance(hsv, cfg)
    ncc, l1 = _gray_symmetry_scores(gray, cfg)

    # 1) single-long 오판정 컷: 채도 균형이 낮으면 무조건 single 취급
    if sat_bal < cfg.sat_balance_threshold:
        return False

    # 2) dual 후보만 명도 대칭 확인
    return (ncc >= cfg.ncc_threshold) and (l1 <= cfg.l1_threshold)


# 기존 테스트/호출부에서 함수명이 이거라면 그대로 유지
def is_dual_sided_timer_cropped(img: Image.Image) -> bool:
    return is_dual_sided_timer_cropped_symmetry(img)
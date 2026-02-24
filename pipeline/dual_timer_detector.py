# dual_timer_detector.py (대칭성 기반 버전)
# - ROI가 "거의 바만" 크롭된(높이 몇 픽셀 수준) 이미지에 최적화
# - 좌/우를 각각 'bar 검출'해서 AND 하는 대신,
#   가운데를 버린 뒤 좌측과 (우측 flip)의 1D 밝기 프로파일이 얼마나 비슷한지로 판정

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image
import cv2


# ======================
# Config
# ======================
@dataclass(frozen=True)
class SymmetryConfig:
    # 중앙 UI(숫자/세로선 등) 영향 제거를 위해 가운데를 버림
    # 예: 0.18이면 중앙 18% 구간을 버림
    center_ignore_ratio: float = 0.148

    # 1D 프로파일 스무딩(노이즈 억제)
    smooth_window: int = 9

    # 대칭 판정 임계값
    # (경험적으로 dual은 높고, single은 낮아짐)
    ncc_threshold: float = 0.70     # 정규화 상관계수(클수록 대칭)
    l1_threshold: float = 0.85      # 정규화 후 평균 절대오차(작을수록 대칭)

    # 너무 “평평한” 이미지(변화 거의 없음)면 대칭성 점수가 의미 없어질 수 있어 안전장치
    min_profile_std: float = 2.0


# ======================
# Utils
# ======================
def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _moving_average_1d(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    win = _clamp_int(int(win), 1, 101)  # 과도한 윈도우 방지
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, k, mode="same").astype(np.float32)


def _zscore_1d(x: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, float]:
    x = x.astype(np.float32)
    m = float(x.mean())
    s = float(x.std())
    return (x - m) / (s + eps), s


def _split_left_right(gray: np.ndarray, center_ignore_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    gray: (H, W) float32/uint8
    return: left_gray, right_gray (각각 H x W')
    """
    h, w = gray.shape
    ci = float(center_ignore_ratio)
    ci = max(0.0, min(ci, 0.6))  # 중앙 버림이 과해지지 않게 제한

    l_end = int(w * (0.5 - ci / 2.0))
    r_start = int(w * (0.5 + ci / 2.0))

    l_end = _clamp_int(l_end, 1, w - 1)
    r_start = _clamp_int(r_start, 1, w - 1)

    left = gray[:, :l_end]
    right = gray[:, r_start:]
    return left, right


def _symmetry_score_profile(gray: np.ndarray, cfg: SymmetryConfig) -> Tuple[float, float]:
    """
    ROI가 매우 얇은(높이 작은) 바 이미지에 강한 방식:
    - 각 절반에서 (세로 평균) 1D 밝기 프로파일을 만듦
    - 우측은 flip해서 좌측과 방향을 맞춤
    - 스무딩 후 z-score 정규화
    - NCC(상관계수)와 L1(평균 절대오차)로 대칭성 평가
    """
    left, right = _split_left_right(gray, cfg.center_ignore_ratio)

    # 세로 평균으로 1D 프로파일 생성
    left_p = left.mean(axis=0).astype(np.float32)
    right_p = right.mean(axis=0).astype(np.float32)

    # 길이 맞추기 (짧은 쪽 기준)
    m = int(min(left_p.shape[0], right_p.shape[0]))
    if m < 10:
        return -1.0, 999.0

    left_p = left_p[:m]
    right_p = right_p[-m:]  # 오른쪽 끝부터 m개
    right_p = right_p[::-1]  # flip

    # 스무딩
    left_p = _moving_average_1d(left_p, cfg.smooth_window)
    right_p = _moving_average_1d(right_p, cfg.smooth_window)

    # 정규화
    left_z, left_std = _zscore_1d(left_p)
    right_z, right_std = _zscore_1d(right_p)

    # “너무 평평한” 경우(표준편차 매우 작음)면 대칭 점수 신뢰도 낮음
    if left_std < cfg.min_profile_std or right_std < cfg.min_profile_std:
        return -1.0, 999.0

    # NCC: [-1, 1], 1에 가까울수록 대칭
    ncc = float(np.mean(left_z * right_z))

    # L1: 작을수록 대칭
    l1 = float(np.mean(np.abs(left_z - right_z)))
    return ncc, l1


# ======================
# Public API
# ======================
def is_dual_sided_timer_cropped_symmetry(
    img: Image.Image,
    cfg: SymmetryConfig = SymmetryConfig(),
) -> bool:
    """
    "바만" 크롭된 ROI 이미지에서 듀얼 타이머(양쪽 대칭 바)인지 판정.
    """
    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)

    ncc, l1 = _symmetry_score_profile(gray, cfg)

    # 듀얼은 보통: ncc 높고, l1 낮음
    return (ncc >= cfg.ncc_threshold) and (l1 <= cfg.l1_threshold)


# ======================
# (선택) 기존 함수명 유지하고 싶으면 아래처럼 래핑해서 교체
# ======================
def is_dual_sided_timer_cropped(img: Image.Image) -> bool:
    return is_dual_sided_timer_cropped_symmetry(img)
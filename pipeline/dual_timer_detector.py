# dual_timer_detector_cropped.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from PIL import Image


@dataclass
class DetectorConfig:
    # 좌/우 분할 비율 (중앙 경계에 UI가 걸칠 수 있어 약간의 안전 마진)
    split_left_end: float = 0.425
    split_right_start: float = 0.575

    # 바는 가로로 긴 연속 구간이어야 함
    min_run_ratio: float = 0.35  # 반쪽 너비 대비 22% 이상 연속 True면 바

    # 적응형 임계값(분위수) + 클램프
    sat_quantile: float = 0.90
    sat_floor: int = 35
    sat_ceiling: int = 170

    val_quantile: float = 0.82
    val_floor: int = 65
    val_ceiling: int = 210

    # 행에서 True 비율이 이 이상이면 "막대가 지나가는 행" 후보
    row_hit_ratio: float = 0.18

    # 노이즈 억제용 스무딩
    smooth_window: int = 9


def _to_hsv_np(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img.convert("HSV"), dtype=np.uint8)


def _long_run_exists(row_bool: np.ndarray, min_run: int) -> bool:
    run = 0
    for v in row_bool:
        if v:
            run += 1
            if run >= min_run:
                return True
        else:
            run = 0
    return False


def _detect_bar(hsv: np.ndarray, cfg: DetectorConfig) -> bool:
    s = hsv[:, :, 1].astype(np.int16)
    v = hsv[:, :, 2].astype(np.int16)

    s_thr = int(np.quantile(s, cfg.sat_quantile))
    s_thr = int(np.clip(s_thr, cfg.sat_floor, cfg.sat_ceiling))

    v_thr = int(np.quantile(v, cfg.val_quantile))
    v_thr = int(np.clip(v_thr, cfg.val_floor, cfg.val_ceiling))

    # 상대적으로 튀는 픽셀 마스크
    mask = (s >= s_thr) & (v >= np.median(v))

    # 행별로 "막대 후보" 행 찾기
    row_ratio = mask.mean(axis=1)
    hits = row_ratio >= cfg.row_hit_ratio

    # 1D smoothing
    if cfg.smooth_window >= 3 and hits.size >= cfg.smooth_window:
        k = cfg.smooth_window
        pad = k // 2
        padded = np.pad(hits.astype(np.int32), (pad, pad), mode="edge")
        sm = np.convolve(padded, np.ones(k, dtype=np.int32), mode="valid")
        hits = sm >= int(np.ceil(k * 0.6))

    if hits.sum() == 0:
        return False

    h, w = mask.shape
    min_run = max(1, int(round(w * cfg.min_run_ratio)))

    for y in np.where(hits)[0]:
        if _long_run_exists(mask[y, :], min_run):
            return True
    return False


def is_dual_sided_timer_cropped(img: Image.Image, cfg: DetectorConfig | None = None) -> bool:
    """
    입력이 '바 부분만 잘린 이미지'일 때:
    좌측 절반에서 바 검출 AND 우측 절반에서 바 검출이면 True
    """
    cfg = cfg or DetectorConfig()
    hsv = _to_hsv_np(img)
    h, w = hsv.shape[:2]

    lx_end = int(round(w * cfg.split_left_end))
    rx_start = int(round(w * cfg.split_right_start))
    lx_end = max(1, min(w - 1, lx_end))
    rx_start = max(lx_end, min(w - 1, rx_start))

    left = hsv[:, :lx_end, :]
    right = hsv[:, rx_start:, :]

    return _detect_bar(left, cfg) and _detect_bar(right, cfg)

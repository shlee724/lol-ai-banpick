# pipeline/prepare_phase_detector.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from pipeline.dual_timer_detector import SymmetryConfig, is_dual_sided_timer_cropped_symmetry


# ======================
# Config
# ======================
@dataclass(frozen=True)
class PreparePhaseConfig:
    # dual_timer_detector(대칭+채도) 쪽 설정이 필요하면 여기서 주입 가능
    dual_cfg: SymmetryConfig = SymmetryConfig()

    # "0초(혹은 0에 준함)"으로 볼 최대값
    # 예: OCR이 "0", "00", "1" 정도로 흔들릴 때 1까지 0 취급 가능
    near_zero_max_seconds: int = 0

    # OCR 결과에서 숫자만 뽑아낼 때 사용
    # 타이머는 보통 2자리(예: 30, 12) 혹은 1~2자리로 뜸
    allow_digits_len_min: int = 1
    allow_digits_len_max: int = 2

    # OCR 불안정할 때를 대비한 “이미지 기반” 보조 판정 사용 여부
    # (OCR이 실패하면 fallback으로 씀)
    use_visual_fallback: bool = True

    # 시각적 fallback에서 "숫자 먹힘(거의 0초 순간)" 판정을 위한 민감도
    # 중앙 숫자 영역에서 엣지/획이 거의 사라진 상태(혹은 너무 단순)면 0초로 판단하는 방식
    fallback_min_edge_density: float = 0.010  # 작을수록 "획이 거의 없음" -> 0에 가깝다고 봄


# ======================
# OCR helpers
# ======================
def _preprocess_digits_for_ocr(img: Image.Image) -> Image.Image:
    """
    숫자 OCR용 전처리:
    - 그레이스케일
    - 확대
    - 대비 강화 + 이진화
    """
    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # 확대(숫자 작으면 OCR 흔들림 완화)
    h, w = gray.shape
    scale = 4
    gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # 블러 -> 적응형 임계(명암 변화에 강함)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )

    return Image.fromarray(bin_img)


def _extract_seconds_from_ocr_text(text: str, cfg: PreparePhaseConfig) -> Optional[int]:
    """
    OCR 텍스트에서 초(second) 숫자만 뽑아 int로 반환.
    """
    if not text:
        return None

    # O/° 같은 오인식 제거용 정규화
    t = text.strip()
    t = t.replace("O", "0").replace("o", "0").replace("°", "0")

    # 숫자만 추출
    digits = re.findall(r"\d+", t)
    if not digits:
        return None

    # 보통 가장 그럴듯한 토큰은 가장 긴 숫자열
    s = max(digits, key=len)

    if not (cfg.allow_digits_len_min <= len(s) <= cfg.allow_digits_len_max):
        return None

    try:
        return int(s)
    except ValueError:
        return None


def _ocr_digits_seconds(digits_img: Image.Image, cfg: PreparePhaseConfig) -> Optional[int]:
    """
    프로젝트의 core/ocr_engine.extract_text를 사용해서 숫자를 읽는다.
    (시그니처가 다를 수 있어 안전하게 try 계열로 처리)
    """
    pre = _preprocess_digits_for_ocr(digits_img)

    try:
        from core.ocr_engine import extract_text  # 네 프로젝트에 이미 존재
    except Exception:
        return None

    text = None

    # 가능한 호출 시그니처를 유연하게 대응
    for kwargs in (
        {"psm": 7, "whitelist": "0123456789"},
        {"psm": 8, "whitelist": "0123456789"},
        {},  # 최후: 기본 호출
    ):
        try:
            text = extract_text(pre, **kwargs)  # type: ignore
            if isinstance(text, str) and text.strip():
                break
        except TypeError:
            # kwargs 지원 안 하면 다음 시도
            continue
        except Exception:
            continue

    if not isinstance(text, str):
        return None

    return _extract_seconds_from_ocr_text(text, cfg)


# ======================
# Visual fallback
# ======================
def _visual_near_zero_fallback(digits_img: Image.Image, cfg: PreparePhaseConfig) -> bool:
    """
    OCR이 실패했을 때만 쓰는 보조 규칙.
    - 중앙 숫자 영역이 '획/윤곽'이 거의 없으면(엣지 밀도 낮음) 0초로 간주.
    """
    rgb = digits_img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # 확대 후 엣지
    h, w = gray.shape
    gray = cv2.resize(gray, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, 40, 120)
    edge_density = float(np.mean(edges > 0))  # 0~1

    return edge_density < cfg.fallback_min_edge_density


# ======================
# Public API
# ======================
def is_timer_near_zero(
    timer_digits_img: Image.Image, cfg: PreparePhaseConfig = PreparePhaseConfig()
) -> bool:
    """
    타이머 중앙 숫자가 0(또는 0에 준함)인지 판정.
    """
    sec = _ocr_digits_seconds(timer_digits_img, cfg)
    if sec is not None:
        return sec <= cfg.near_zero_max_seconds

    if cfg.use_visual_fallback:
        return _visual_near_zero_fallback(timer_digits_img, cfg)

    return False


def is_dual_timer_effective(
    timer_bar_img: Image.Image,
    timer_digits_img: Image.Image,
    cfg: PreparePhaseConfig = PreparePhaseConfig(),
) -> bool:
    """
    1) 바 대칭(dual) 판정
    2) BUT 타이머가 0초(혹은 0 근처)면 dual/single이 똑같이 보이므로
       무조건 False로 강제(= single 취급)
    """
    if is_timer_near_zero(timer_digits_img, cfg):
        return False

    return is_dual_sided_timer_cropped_symmetry(timer_bar_img, cfg.dual_cfg)

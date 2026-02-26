from PIL import Image

from pipeline.prepare_phase_detector import (
    is_dual_timer_effective,
    is_timer_near_zero,
)


# ----------------------------
# 유틸: 더미 이미지 생성
# ----------------------------
def make_dummy_img(w=200, h=40, color=(0, 0, 0)):
    return Image.new("RGB", (w, h), color)


# ----------------------------
# OCR 모킹용 helper
# ----------------------------
class DummyOCR:
    def __init__(self, value):
        self.value = value

    def __call__(self, *args, **kwargs):
        return self.value


# ----------------------------
# is_timer_near_zero 테스트
# ----------------------------


def test_timer_near_zero_when_ocr_returns_zero(monkeypatch):
    """
    OCR 결과가 "0"이면 near_zero True
    """
    from pipeline import prepare_phase_detector as mod

    monkeypatch.setattr(mod, "_ocr_digits_seconds", lambda img, cfg: 0)

    img = make_dummy_img()
    assert is_timer_near_zero(img) is True


def test_timer_not_near_zero_when_ocr_returns_positive(monkeypatch):
    """
    OCR 결과가 5초면 near_zero False
    """
    from pipeline import prepare_phase_detector as mod

    monkeypatch.setattr(mod, "_ocr_digits_seconds", lambda img, cfg: 5)

    img = make_dummy_img()
    assert is_timer_near_zero(img) is False


def test_timer_near_zero_when_ocr_fails_and_fallback_true(monkeypatch):
    """
    OCR 실패(None) + fallback True → near_zero True
    """
    from pipeline import prepare_phase_detector as mod

    monkeypatch.setattr(mod, "_ocr_digits_seconds", lambda img, cfg: None)
    monkeypatch.setattr(mod, "_visual_near_zero_fallback", lambda img, cfg: True)

    img = make_dummy_img()
    assert is_timer_near_zero(img) is True


def test_timer_not_near_zero_when_ocr_fails_and_fallback_false(monkeypatch):
    """
    OCR 실패(None) + fallback False → near_zero False
    """
    from pipeline import prepare_phase_detector as mod

    monkeypatch.setattr(mod, "_ocr_digits_seconds", lambda img, cfg: None)
    monkeypatch.setattr(mod, "_visual_near_zero_fallback", lambda img, cfg: False)

    img = make_dummy_img()
    assert is_timer_near_zero(img) is False


# ----------------------------
# is_dual_timer_effective 테스트
# ----------------------------


def test_force_single_when_timer_zero(monkeypatch):
    """
    0초 순간이면 dual 판정이 True여도 False로 강제해야 함
    """
    from pipeline import prepare_phase_detector as mod

    # 타이머는 0초
    monkeypatch.setattr(mod, "is_timer_near_zero", lambda img, cfg=None: True)

    # dual_timer_detector는 True라고 가정
    monkeypatch.setattr(mod, "is_dual_sided_timer_cropped_symmetry", lambda img, cfg=None: True)

    bar_img = make_dummy_img()
    digits_img = make_dummy_img()

    assert is_dual_timer_effective(bar_img, digits_img) is False


def test_dual_when_not_zero_and_symmetry_true(monkeypatch):
    """
    0초 아니고 dual symmetry True면 True
    """
    from pipeline import prepare_phase_detector as mod

    monkeypatch.setattr(mod, "is_timer_near_zero", lambda img, cfg=None: False)
    monkeypatch.setattr(mod, "is_dual_sided_timer_cropped_symmetry", lambda img, cfg=None: True)

    bar_img = make_dummy_img()
    digits_img = make_dummy_img()

    assert is_dual_timer_effective(bar_img, digits_img) is True


def test_single_when_not_zero_and_symmetry_false(monkeypatch):
    """
    0초 아니고 symmetry False면 False
    """
    from pipeline import prepare_phase_detector as mod

    monkeypatch.setattr(mod, "is_timer_near_zero", lambda img, cfg=None: False)
    monkeypatch.setattr(mod, "is_dual_sided_timer_cropped_symmetry", lambda img, cfg=None: False)

    bar_img = make_dummy_img()
    digits_img = make_dummy_img()

    assert is_dual_timer_effective(bar_img, digits_img) is False

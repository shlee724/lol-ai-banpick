from __future__ import annotations

from typing import Optional

import pytest
from PIL import Image

from config.path import PATHS
from pipeline.dual_timer_detector import is_dual_sided_timer_cropped  # 실제 엔트리 함수명에 맞게



def _infer_expected_from_name(name: str) -> Optional[bool]:
    n = name.lower()

    if any(k in n for k in ["dual", "both", "양쪽"]):
        return True

    if any(k in n for k in ["single", "한쪽"]):
        return False

    return None


def _iter_images():
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    test_case = "test_4"
    test_dir = PATHS.TEST_BANPICK_TIMER_DIR / test_case
    for p in sorted(test_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


@pytest.mark.parametrize("img_path", list(_iter_images()))
def test_dual_timer_detector(img_path):
    expected = _infer_expected_from_name(img_path.name)

    if expected is None:
        pytest.skip(f"라벨 없는 샘플 스킵: {img_path.name}")

    img = Image.open(img_path)
    result = is_dual_sided_timer_cropped(img)

    assert result == expected, f"{img_path.name}: expected={expected}, got={result}"
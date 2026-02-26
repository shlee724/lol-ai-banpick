# pipeline/pick_stage_detector.py
from dataclasses import dataclass

from PIL import Image

from pipeline.ban_detector import detect_ban_strip_variance


def merge_images_horizontal(
    img1: Image.Image, img2: Image.Image, bg_color=(255, 255, 255)
) -> Image.Image:
    # tests/test_rois.py와 동일한 방식 :contentReference[oaicite:1]{index=1}
    new_width = img1.width + img2.width
    new_height = max(img1.height, img2.height)

    new_img = Image.new("RGB", (new_width, new_height), bg_color)
    new_img.paste(img1.convert("RGB"), (0, 0))
    new_img.paste(img2.convert("RGB"), (img1.width, 0))
    return new_img


@dataclass
class PickStageResult:
    kind: str  # "PICK_REAL" | "PICK_FAKE"
    std: float


def detect_pick_kind_from_banned_strips(
    my_banned_strip: Image.Image,
    enemy_banned_strip: Image.Image,
    *,
    std_threshold: float = 18.0,
) -> PickStageResult:
    total_banned = merge_images_horizontal(my_banned_strip, enemy_banned_strip)
    res = detect_ban_strip_variance(total_banned, std_threshold=std_threshold)

    kind = "PICK_REAL" if res.is_filled else "PICK_FAKE"
    return PickStageResult(kind=kind, std=res.std)

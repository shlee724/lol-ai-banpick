from PIL import Image


def crop_roi_definite_xy(img: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    """
    img: 전체 캡처 이미지
    x, y: 좌상단 기준 좌표
    w, h: ROI 크기
    """
    return img.crop((x, y, x + w, y + h))


def crop_roi_relative_xy(
    img: Image.Image, window_size: tuple[int, int], roi: tuple[float, float, float, float]
) -> Image.Image:
    """
    img: 전체 캡처 이미지
    window_size: 전체 캡처 이미지의 사이즈 (w, h)
    roi: 원본 이미지 기준 상대적인 위치 (x, y, w, h)
    """
    o_w, o_h = window_size
    r_x, r_y, r_w, r_h = roi
    target_x = o_w * r_x
    target_y = o_h * r_y
    target_w = o_w * r_w
    target_h = o_h * r_h

    return img.crop((target_x, target_y, target_x + target_w, target_y + target_h))

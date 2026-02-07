# core/screen_capture.py

import win32gui
import win32ui
import win32con
import ctypes
import numpy as np
from PIL import Image
from typing import Tuple


def capture_window(hwnd: int, width: int, height: int) -> Image.Image:
    """
    hwnd: 롤 클라이언트 윈도우 핸들
    """
    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()

    save_bitmap = win32ui.CreateBitmap()
    save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
    save_dc.SelectObject(save_bitmap)

    # PrintWindow (DX 대응)
    result = ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)
    if result != 1:
        raise RuntimeError("PrintWindow failed")

    bmp_info = save_bitmap.GetInfo()
    bmp_str = save_bitmap.GetBitmapBits(True)

    img = np.frombuffer(bmp_str, dtype=np.uint8)
    img.shape = (bmp_info['bmHeight'], bmp_info['bmWidth'], 4)

    img = img[:, :, :3][:, :, ::-1]  # BGRA → RGB

    # 리소스 해제
    win32gui.DeleteObject(save_bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)

    return Image.fromarray(img)

def crop_roi(img: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    """
    img: 전체 캡처 이미지
    x, y: 좌상단 기준 좌표
    w, h: ROI 크기
    """
    return img.crop((x, y, x + w, y + h))

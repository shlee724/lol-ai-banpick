# core/ocr_engine.py

import pytesseract
import cv2
import numpy as np
from PIL import Image


def preprocess_for_ocr(pil_img: Image.Image):
    """
    OCR 정확도 향상을 위한 전처리
    """
    img = np.array(pil_img)

    # RGB → GRAY
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 노이즈 제거
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 이진화
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def extract_text(pil_img: Image.Image) -> str:
    processed = preprocess_for_ocr(pil_img)

    text = pytesseract.image_to_string(
        processed,
        lang="kor",
        config="--psm 6"
    )

    return text.strip()

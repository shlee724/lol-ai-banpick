import re

class TextNormalizer:
    def __init__(self):
        # 자주 깨지는 OCR 문자 치환 규칙
        self.replace_map = {
            "챔피언을 선태하세요": "챔피언을 선택하세요",
            "신택": "선택",
            "진핵": "선택",
            "신웅": "선택",
            "잼피인": "챔피언",
        }

    def normalize(self, text: str) -> str:
        if not text:
            return ""

        t = text.strip()
        t = t.replace("\n", " ").replace("\r", " ")
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"[^0-9가-힣a-zA-Z !?.]", "", t)

        for k, v in self.replace_map.items():
            t = t.replace(k, v)

        return t.strip()
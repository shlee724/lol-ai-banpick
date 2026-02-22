# core/gemini_vision.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Optional
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from typing import Iterator

@dataclass
class GeminiVisionResult:
    text: str
    raw: Any  # response 객체(디버그용)


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def get_client() -> genai.Client:
    load_dotenv("config/.env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 .env에 없습니다.")
    return genai.Client(api_key=api_key)


def analyze_image(
    img: Image.Image,
    *,
    prompt: str,
    model: str = "gemini-2.0-flash",
    mime_type: str = "image/png",
) -> GeminiVisionResult:
    """
    PIL Image 1장을 Gemini에 보내고 response.text를 반환.
    """
    client = get_client()
    img_bytes = _pil_to_png_bytes(img)

    response = client.models.generate_content(
        model=model,
        contents=[
            prompt,
            types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
        ],
    )

    return GeminiVisionResult(text=(response.text or "").strip(), raw=response)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL | re.IGNORECASE)

def _extract_json_text(text: str) -> str:
    """
    Gemini가 ```json ... ``` 형태로 내놓거나 앞뒤에 잡텍스트를 붙여도
    JSON 부분만 최대한 안전하게 뽑아낸다.
    """
    t = (text or "").strip()

    # 1) ```json ... ``` 펜스 우선 추출
    m = _JSON_FENCE_RE.search(t)
    if m:
        return m.group(1).strip()

    # 2) 펜스가 없어도, 텍스트 중 첫 '{'부터 마지막 '}'까지 잘라보기
    if "{" in t and "}" in t:
        start = t.find("{")
        end = t.rfind("}")
        if start < end:
            return t[start:end + 1].strip()

    # 3) 리스트 JSON일 수도 있으니 []도 처리
    if "[" in t and "]" in t:
        start = t.find("[")
        end = t.rfind("]")
        if start < end:
            return t[start:end + 1].strip()

    return t


def analyze_image_json(
    img: Image.Image,
    *,
    prompt: str,
    model: str = "gemini-2.0-flash",
    mime_type: str = "image/png",
) -> Dict[str, Any]:
    json_only_prompt = prompt.strip() + "\n\nReturn ONLY valid JSON. No markdown. No extra text."
    res = analyze_image(img, prompt=json_only_prompt, model=model, mime_type=mime_type)

    cleaned = _extract_json_text(res.text)

    try:
        return json.loads(cleaned)
    except Exception as e:
        raise RuntimeError(
            f"Gemini JSON 파싱 실패: {e}\n"
            f"--- cleaned ---\n{cleaned}\n"
            f"--- raw text ---\n{res.text}"
        )

def analyze_image_stream(
    img: Image.Image,
    *,
    prompt: str,
    model: str = "gemini-2.0-flash",
    mime_type: str = "image/png",
    config: Optional[types.GenerateContentConfig] = None,
) -> Iterator[str]:
    """
    Gemini 스트리밍 호출.
    yield 되는 문자열은 '추가로 생성된 텍스트 조각'이다.
    """
    client = get_client()
    img_bytes = _pil_to_png_bytes(img)

    stream = client.models.generate_content_stream(
        model=model,
        contents=[
            prompt,
            types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
        ],
        config=config,
    )

    for chunk in stream:
        # chunk.text가 None인 경우가 있을 수 있어 방어
        t = (chunk.text or "")
        if t:
            yield t
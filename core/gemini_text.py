# core/gemini_text.py
from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from google import genai


@dataclass
class GeminiTextResult:
    text: str
    raw: Any  # response 객체(디버그용)


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

    # 2) 객체 JSON: 첫 '{'부터 마지막 '}'까지
    if "{" in t and "}" in t:
        start = t.find("{")
        end = t.rfind("}")
        if start < end:
            return t[start:end + 1].strip()

    # 3) 리스트 JSON: 첫 '['부터 마지막 ']'까지
    if "[" in t and "]" in t:
        start = t.find("[")
        end = t.rfind("]")
        if start < end:
            return t[start:end + 1].strip()

    return t


def get_client() -> genai.Client:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 .env에 없습니다.")
    return genai.Client(api_key=api_key)


def generate_text(
    prompt: str,
    *,
    model: str = "gemini-2.0-flash",
) -> GeminiTextResult:
    """
    텍스트 프롬프트만 보내고 response.text를 반환.
    """
    client = get_client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return GeminiTextResult(text=(response.text or "").strip(), raw=response)


def generate_text_json(
    prompt: str,
    *,
    model: str = "gemini-2.0-flash",
) -> Dict[str, Any]:
    """
    텍스트 프롬프트만 보내고 JSON(dict)로 파싱해서 반환.
    (코드펜스/잡텍스트 자동 제거)
    """
    json_only_prompt = prompt.strip() + "\n\nReturn ONLY valid JSON. No markdown. No extra text."
    res = generate_text(json_only_prompt, model=model)

    cleaned = _extract_json_text(res.text)
    try:
        return json.loads(cleaned)
    except Exception as e:
        raise RuntimeError(
            f"Gemini JSON 파싱 실패: {e}\n"
            f"--- cleaned ---\n{cleaned}\n"
            f"--- raw text ---\n{res.text}"
        )

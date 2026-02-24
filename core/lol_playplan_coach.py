from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterator, Optional, Union, Tuple, List

from dotenv import load_dotenv
from google import genai
from google.genai import types

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


# =========================
# 내부 프롬프트 (현재는 미드 기준)
# =========================
_PROMPT_LOL_PLAYPLAN = """너는 LoL(리그 오브 레전드) '미드' 전문 코치다.
첨부된 이미지는 밴픽 화면의 "픽된 챔피언 정보"가 보이는 이미지다.

목표:
1) 이미지에서 우리 팀(최대 5) / 상대 팀(최대 5)의 '픽된 챔피언'을 가능한 한 정확히 읽는다.
2) 그 조합을 근거로, "우리 팀 미드가 해야 할 플레이 플랜"을 한국어로 제시한다.

중요 규칙:
- 이미지에서 확실히 읽히지 않는 챔피언은 임의 생성 금지.
- 보이지 않는 정보(밴, 룬, 소환사 주문 등) 추측 금지.

출력 형식 (텍스트만):
1) 우리 팀 픽: <챔프1>, <챔프2>, ...
2) 상대 팀 픽: <챔프1>, <챔프2>, ...
3) 플레이 플랜은 승리를 위해 중요한 점 위주로 상세하게
4) 말투는 존댓말로
"""


# =========================
# Client 싱글턴
# =========================
_client_singleton: Optional[genai.Client] = None


def get_playplan_coach_client(api_key_env: str = "GEMINI_API_KEY") -> genai.Client:
    return _get_client(api_key_env)


def _get_client(api_key_env: str) -> genai.Client:
    global _client_singleton
    if _client_singleton is not None:
        return _client_singleton

    load_dotenv("config/.env")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} 환경변수가 비어 있습니다.")

    _client_singleton = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=30_000,
            retry_options=types.HttpRetryOptions(
                attempts=1,
                initial_delay=0.5,
                max_delay=2.0,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            ),
        ),
    )
    return _client_singleton


InputImage = Union["Image.Image", bytes, str, Path]


def _to_image_bytes(inp: InputImage, *, mime_type: str) -> bytes:
    if isinstance(inp, (str, Path)):
        return Path(inp).read_bytes()

    if isinstance(inp, (bytes, bytearray)):
        return bytes(inp)

    if Image is None:
        raise RuntimeError("PIL이 설치되어 있지 않습니다.")

    img: "Image.Image" = inp  # type: ignore
    from io import BytesIO

    buf = BytesIO()
    if mime_type == "image/png":
        img.save(buf, format="PNG")
    elif mime_type in ("image/jpeg", "image/jpg"):
        img.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
    else:
        raise ValueError(f"지원하지 않는 mime_type: {mime_type}")

    return buf.getvalue()


# =========================
# 스트리밍 함수
# =========================
def lol_playplan_stream(
    picked_champs_img: InputImage,
    *,
    client: Optional[genai.Client] = None,
    model: str = "gemini-2.5-pro",
    mime_type: str = "image/png",
    temperature: float = 0.2,
    max_output_tokens: int = 2000,
    thinking_budget: int = 256,
    api_key_env: str = "GEMINI_API_KEY",
) -> Iterator[str]:

    if client is None:
        client = _get_client(api_key_env)

    img_bytes = _to_image_bytes(picked_champs_img, mime_type=mime_type)

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=_PROMPT_LOL_PLAYPLAN),
                types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
            ],
        )
    ]

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
    )

    stream = client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    )

    for chunk in stream:
        text = chunk.text or ""
        if text:
            yield text


# =========================
# 논스트리밍 래퍼
# =========================
def lol_playplan_run(
    picked_champs_img: InputImage,
    *,
    client: Optional[genai.Client] = None,
    model: str = "gemini-2.5-pro",
    mime_type: str = "image/png",
    temperature: float = 0.2,
    max_output_tokens: int = 600,
    thinking_budget: int = 256,
    api_key_env: str = "GEMINI_API_KEY",
    return_timings: bool = True,
) -> Union[str, Tuple[str, float, float]]:

    t0 = time.perf_counter()
    first_token_s: Optional[float] = None
    buf: List[str] = []

    for delta in lol_playplan_stream(
        picked_champs_img,
        client=client,
        model=model,
        mime_type=mime_type,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_budget=thinking_budget,
        api_key_env=api_key_env,
    ):
        if first_token_s is None:
            first_token_s = time.perf_counter() - t0
        buf.append(delta)

    total_s = time.perf_counter() - t0
    text = "".join(buf)

    if not return_timings:
        return text

    return text, (first_token_s or total_s), total_s
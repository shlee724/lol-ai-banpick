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
# 내부 프롬프트(원본 그대로 내장)
# =========================
_PROMPT_LOL_MID_COACH = """너는 LoL 밴픽 코치다. 첨부된 이미지는 "지금 내 픽 차례"인 밴픽 화면이다.
이미지에서 우리/상대 픽과 밴 정보를 읽고, 내가 할 '미드 픽' 3개를 추천한다.

출력 규칙:
- 추천 3개만 출력한다.
- 각 추천은 정확히 2줄로만 출력한다:
  1) <챔피언> - <점수>/10
  2) 라인전: <핵심근거 1개> | 팀가치: <핵심근거 1개>
- 추가 설명, 요약, 머리말 금지.

평가 원칙:
- 픽된 상대 챔피언들 중 미드로 올 가능성이 높은 챔피언들 상대로 라인전 상성이 가장 중요하다. 다만, 픽된 챔피언들 중 미드로 올 가능성이 높은 챔피언이 없다면 선픽으로 무난한 챔피언을 추천한다.
- 팀가치(운영/교전/한타)는 라인전 판단 이후 반영.
- 후보 풀 안에 있는 괄호는 플레이어의 숙련도 점수이다. 플레이어의 숙련도가 3순위 판단기준.
- 추천 챔피언은 후보 풀에서만 고른다.
- 챔피언 이름은 공식 영어명으로만 쓴다.

후보 풀:
[오리아나(39166), 말자하(30972), 갈리오(33888), 말파이트(57893), 문도(63136), 가렌(65600), 초가스(54983), 나서스(88356), 사이온(15886), 하이머딩거(29350), 사일러스(31145), 요네(13998), 레넥톤(24679) ]
"""

# =========================
# Client 싱글턴 (매번 만들지 말고 재사용)
# =========================
_client_singleton: Optional[genai.Client] = None

def get_client(api_key_env: str = "GEMINI_API_KEY") -> genai.Client:
    """외부에서 미리 client를 준비(워밍업)할 수 있게 공개 함수로 제공."""
    return _get_client(api_key_env=api_key_env)

def _get_client(api_key_env: str = "GEMINI_API_KEY") -> genai.Client:
    global _client_singleton
    if _client_singleton is not None:
        return _client_singleton

    load_dotenv("config/.env")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"환경변수 {api_key_env} 가 비어 있습니다. .env 또는 환경변수를 확인하세요.")

    # 원본 코드의 http_options를 그대로 사용 :contentReference[oaicite:1]{index=1}
    _client_singleton = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=20_000,
            retry_options=types.HttpRetryOptions(
                attempts=1,
                initial_delay=0.2,
                max_delay=0.5,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            ),
        ),
    )
    return _client_singleton


InputImage = Union["Image.Image", bytes, str, Path]


def _to_image_bytes(inp: InputImage, *, mime_type: str) -> bytes:
    """
    입력을 bytes로 정규화한다.
    - Path/str: 파일 read_bytes()
    - bytes: 그대로
    - PIL Image: 지정된 mime_type으로 인코딩
    """
    if isinstance(inp, (str, Path)):
        return Path(inp).read_bytes()
    if isinstance(inp, (bytes, bytearray)):
        return bytes(inp)

    if Image is None:
        raise RuntimeError("PIL(Image)이 설치되어 있지 않은데 Image.Image 입력이 들어왔습니다.")

    # PIL Image -> bytes
    img: "Image.Image" = inp  # type: ignore[assignment]

    from io import BytesIO

    buf = BytesIO()
    if mime_type == "image/png":
        # PNG 기본 저장(원하면 compress_level 조절 가능)
        img.save(buf, format="PNG")
    elif mime_type in ("image/jpeg", "image/jpg"):
        img.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
    else:
        raise ValueError(f"지원하지 않는 mime_type: {mime_type}")

    return buf.getvalue()


def lol_mid_pick_coach_stream(
    total_picked_img: InputImage,
    *,
    client: Optional[genai.Client] = None,
    model: str = "gemini-2.5-pro",
    mime_type: str = "image/png",
    temperature: float = 0.2,
    max_output_tokens: int = 500,
    thinking_budget: int = 256,
    api_key_env: str = "GEMINI_API_KEY",
) -> Iterator[str]:
    """
    (스트리밍) 밴픽 이미지 1장 -> 미드 픽 3개 추천 텍스트를 chunk 단위로 yield.

    - 프롬프트는 함수 내부에 '내장'되어 있음(나중에 분리 가능)
    - 입력은 PIL Image / bytes / 파일경로(str|Path) 지원
    """
    # ✅ client 주입되면 그걸 사용, 없으면 기존처럼 싱글턴 생성
    if client is None:
        client = _get_client(api_key_env=api_key_env)
    img_bytes = _to_image_bytes(total_picked_img, mime_type=mime_type)

    # 원본 코드와 동일하게 정식 Content/Part 구성 :contentReference[oaicite:2]{index=2}
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=_PROMPT_LOL_MID_COACH),
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
        t = chunk.text or ""
        if t:
            yield t


def lol_mid_pick_coach_run(
    total_picked_img: InputImage,
    *,
    model: str = "gemini-2.5-pro",
    mime_type: str = "image/png",
    temperature: float = 0.2,
    max_output_tokens: int = 300,
    thinking_budget: int = 128,
    api_key_env: str = "GEMINI_API_KEY",
    return_timings: bool = True,
) -> Union[str, Tuple[str, float, float]]:
    """
    (논스트리밍 래퍼) 내부적으로 stream을 끝까지 모아서 최종 문자열을 반환.
    return_timings=True면 (text, first_token_s, total_s) 반환.
    """
    t0 = time.perf_counter()
    first_token_s: Optional[float] = None
    buf: List[str] = []

    for delta in lol_mid_pick_coach_stream(
        total_picked_img,
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
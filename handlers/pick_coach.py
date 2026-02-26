from __future__ import annotations

import time
from typing import Iterable, Optional

from core.lol_pick_coach import lol_mid_pick_coach_stream


def run_streaming(label: str, stream_iter: Iterable[str]) -> str:
    chunks: list[str] = []
    start_t = time.perf_counter()
    first_token_time: Optional[float] = None

    for delta in stream_iter:
        if first_token_time is None:
            first_token_time = time.perf_counter()
            print(f"\n[{label}] ⏱ 첫 토큰: {first_token_time - start_t:.2f}s\n")
        print(delta, end="", flush=True)
        chunks.append(delta)

    end_t = time.perf_counter()
    print(f"\n\n[{label}] ⏱ 전체: {end_t - start_t:.2f}s")
    return "".join(chunks)


def run_pick_coach(picks_merged_img, *, client: object, model: str) -> str:
    return run_streaming(
        "PICK_COACH",
        lol_mid_pick_coach_stream(
            picks_merged_img,
            client=client,
            model=model,
        ),
    )
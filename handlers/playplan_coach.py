from __future__ import annotations

from core.lol_playplan_coach import lol_playplan_stream
from handlers.pick_coach import run_streaming  # 동일 스트리밍 출력 재사용


def run_playplan(picks_merged_img, *, client: object, model: str) -> str:
    return run_streaming(
        "PLAYPLAN_COACH",
        lol_playplan_stream(
            picks_merged_img,
            client=client,
            model=model,
        ),
    )
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

CommandType = Literal[
    "RESET_DUAL_BUF",
    "PUSH_DUAL_NOW",
    "CALL_PICK_COACH",
    "CALL_PLAYPLAN",
    "STOP",
]

@dataclass(frozen=True)
class Command:
    type: CommandType
    # 공용 페이로드(필요할 때만 사용)
    dual_now: Optional[bool] = None
    reason: str = ""
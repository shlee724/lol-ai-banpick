from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.frame_types import FrameContext


@dataclass(frozen=True)
class WorkflowResult:
    should_stop: bool = False


class Workflow(Protocol):
    def on_frame(self, ctx: FrameContext) -> WorkflowResult: ...
from __future__ import annotations
from typing import Protocol, Sequence

from app.frame_types import FrameContext
from workflow.commands import Command

class Workflow(Protocol):
    def on_frame(self, ctx: FrameContext) -> Sequence[Command]:
        ...
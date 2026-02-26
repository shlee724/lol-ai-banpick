from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

from app.frame_types import FrameContext
from app.wiring import AppDeps
from app.settings import AppSettings

from workflow.commands import Command

from handlers.pick_coach import run_pick_coach
from handlers.playplan_coach import run_playplan


@dataclass
class CommandRunner:
    settings: AppSettings
    deps: AppDeps

    def execute(self, ctx: FrameContext, commands: Sequence[Command]) -> bool:
        """
        return True if should_stop
        """
        for cmd in commands:
            t = cmd.type

            if t == "RESET_DUAL_BUF":
                self.deps.dual_buf.reset()

            elif t == "PUSH_DUAL_NOW":
                if cmd.dual_now is None:
                    continue
                self.deps.dual_buf.push(cmd.dual_now)
                dual_stable = self.deps.dual_buf.get_majority()
                dual_conf = self.deps.dual_buf.get_confidence()

                print(f"[PREPARE] DualEffective: now={cmd.dual_now} stable={dual_stable} ({dual_conf:.2f})")

                if dual_stable is True and dual_conf >= self.settings.dual_conf_threshold:
                    print("[PREPARE] 양팀 모든 챔피언 픽 됐습니다 (stable)")
                    run_playplan(
                        ctx.rois.picks_merged_img,
                        client=self.deps.playplan_client,  # wiring에서 노출 필요(아래 참고)
                        model=self.settings.gemini_model,
                    )
                    return True

            elif t == "CALL_PICK_COACH":
                run_pick_coach(
                    ctx.rois.picks_merged_img,
                    client=self.deps.pick_client,  # wiring에서 노출 필요
                    model=self.settings.gemini_model,
                )

            elif t == "CALL_PLAYPLAN":
                run_playplan(
                    ctx.rois.picks_merged_img,
                    client=self.deps.playplan_client,
                    model=self.settings.gemini_model,
                )
                return True

            elif t == "STOP":
                return True

        return False
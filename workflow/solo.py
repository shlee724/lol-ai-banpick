from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, List

from app.frame_types import FrameContext
from workflow.commands import Command

from pipeline.pick_stage_detector import detect_pick_kind_from_banned_strips
from pipeline.prepare_phase_detector import is_dual_timer_effective


@dataclass
class SoloWorkflow:
    pick_std_threshold: float
    dual_conf_threshold: float

    pick_real_executed: bool = False

    def on_frame(self, ctx: FrameContext) -> Sequence[Command]:
        s = ctx.state
        r = ctx.rois

        cmds: List[Command] = []

        if s.stable_state == "PICK":
            # PREPARE 아니므로 dual 안정화 버퍼 리셋(원래 main.py의 else: dual_buf.reset)
            cmds.append(Command(type="RESET_DUAL_BUF"))

            # Guards (원래 main.py 동일)
            if s.raw_state == "BAN":
                return cmds
            if self.pick_real_executed:
                return cmds

            pick_res = detect_pick_kind_from_banned_strips(
                r.bans_my_img,
                r.bans_enemy_img,
                std_threshold=self.pick_std_threshold,
            )
            print(f"[PICK] 판정: kind={pick_res.kind} std={pick_res.std:.2f}")

            if pick_res.kind == "PICK_REAL":
                # 실제 호출은 runner가 함
                cmds.append(Command(type="CALL_PICK_COACH"))
                self.pick_real_executed = True

            return cmds

        if s.stable_state == "PREPARE":
            dual_now = is_dual_timer_effective(
                timer_bar_img=r.timer_bar_img,
                timer_digits_img=r.timer_digits_img,
            )
            cmds.append(Command(type="PUSH_DUAL_NOW", dual_now=dual_now))
            return cmds

        # 그 외 상태: dual_buf.reset (원래 main.py의 else)
        cmds.append(Command(type="RESET_DUAL_BUF"))
        return cmds
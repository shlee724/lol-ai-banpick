from __future__ import annotations

from dataclasses import dataclass

from app.frame_types import FrameContext
from workflow.base import WorkflowResult

from pipeline.pick_stage_detector import detect_pick_kind_from_banned_strips
from pipeline.prepare_phase_detector import is_dual_timer_effective

from handlers.pick_coach import run_pick_coach
from handlers.playplan_coach import run_playplan


@dataclass
class SoloWorkflow:
    pick_coach_client: object
    playplan_client: object
    gemini_model: str
    pick_std_threshold: float
    dual_conf_threshold: float

    # 상태
    pick_real_executed: bool = False
    # dual 안정화는 run.py에서 공용 버퍼로 관리하지만,
    # "현재 프레임 dual_now" 계산은 workflow가 해도 됨.
    # (원래 main.py에서 PREPARE 분기 안에서 계산했기 때문)

    def on_pick(self, ctx: FrameContext) -> WorkflowResult:
        rois = ctx.rois
        state = ctx.state

        # Guards (원래 main.py 동일)
        if state.raw_state == "BAN":
            return WorkflowResult()

        if self.pick_real_executed:
            return WorkflowResult()

        pick_res = detect_pick_kind_from_banned_strips(
            rois.bans_my_img,
            rois.bans_enemy_img,
            std_threshold=self.pick_std_threshold,
        )
        print(f"[PICK] 판정: kind={pick_res.kind} std={pick_res.std:.2f}")

        if pick_res.kind == "PICK_REAL":
            try:
                run_pick_coach(
                    rois.picks_merged_img,
                    client=self.pick_coach_client,
                    model=self.gemini_model,
                )
                self.pick_real_executed = True
            except Exception as e:
                print("[ERR] Gemini 호출 실패:", repr(e))

        return WorkflowResult()

    def get_dual_now(self, ctx: FrameContext) -> bool:
        rois = ctx.rois
        return is_dual_timer_effective(
            timer_bar_img=rois.timer_bar_img,
            timer_digits_img=rois.timer_digits_img,
        )

    def on_prepare_dual_stable(self, ctx: FrameContext) -> WorkflowResult:
        print("[PREPARE] 양팀 모든 챔피언 픽 됐습니다 (stable)")

        run_playplan(
            ctx.rois.picks_merged_img,
            client=self.playplan_client,
            model=self.gemini_model,
        )
        return WorkflowResult(should_stop=True)
from dataclasses import dataclass
# x, y, w, h
# 원본 이미지(롤 클라이언트) 기준 상대적 왼쪽 위 모서리의 x, 왼쪽 위 모서리의 y, 너비 w, 높이 h
@dataclass(frozen=True)
class ROISet:
    BANPICK_STATUS_TEXT = (0.28, 0.00, 0.44, 0.08)
    BANPICK_TIMER_BAR = (0.28, 0.08, 0.44, 0.02)

    BANNED_CHAMPIONS_CENTER = (0.29, 0.16, 0.42, 0.59)
    BANNED_CHAMPIONS_MY_TEAM = (0.01, 0.04, 0.15, 0.045)
    BANNED_CHAMPIONS_ENEMY_TEAM = (0.84, 0.04, 0.15, 0.045)

    PICKED_CHAMPIONS_MY_TEAM = (0.00, 0.129, 0.25, 0.57)
    PICKED_CHAMPIONS_ENEMY_TEAM = (0.75, 0.129, 0.25, 0.57)

ROI = ROISet()

import time

class StableStateManager:
    def __init__(self, min_duration=1.0, min_confidence=0.6):
        self.current_state = None
        self.last_change_time = time.time()
        self.min_duration = min_duration
        self.min_confidence = min_confidence

    def update(self, candidate_state: str, confidence: float):
        now = time.time()

        if self.current_state is None:
            self.current_state = candidate_state
            self.last_change_time = now
            return self.current_state

        # 동일 상태 유지
        if candidate_state == self.current_state:
            return self.current_state

        # 신뢰도 조건
        if confidence < self.min_confidence:
            return self.current_state

        # 시간 조건
        if now - self.last_change_time < self.min_duration:
            return self.current_state

        # 상태 변경 승인
        self.current_state = candidate_state
        self.last_change_time = now
        return self.current_state
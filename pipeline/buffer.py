from collections import Counter, deque


class StateBuffer:
    def __init__(self, size: int = 7):
        self.buffer = deque(maxlen=size)

    def reset(self) -> None:
        self.buffer.clear()

    def push(self, state: str):
        self.buffer.append(state)

    def get_majority(self) -> str:
        if not self.buffer:
            return "UNKNOWN"

        counter = Counter(self.buffer)
        state, count = counter.most_common(1)[0]
        return state

    def get_confidence(self) -> float:
        if not self.buffer:
            return 0.0

        counter = Counter(self.buffer)
        state, count = counter.most_common(1)[0]
        return count / len(self.buffer)

class StateClassifier:
    def __init__(self):
        self.rules = {
            "PICK": ["선택하세요"],
            "BAN": ["금지할", "밴"],
            "PREPARE": ["장비를 준비하세요"],
            "FIGHT": ["전투 준비"],
        }

    def classify(self, text: str) -> str:
        if not text:
            return "UNKNOWN"

        for state, keywords in self.rules.items():
            for kw in keywords:
                if kw in text:
                    return state

        return "UNKNOWN"
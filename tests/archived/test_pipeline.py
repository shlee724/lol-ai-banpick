from pathlib import Path

from PIL import Image

from config.path import PATHS
from core.ocr_engine import extract_text
from pipeline.buffer import StateBuffer
from pipeline.classifier import StateClassifier
from pipeline.normalizer import TextNormalizer
from pipeline.state_manager import StableStateManager

normalizer = TextNormalizer()
classifier = StateClassifier()
buffer = StateBuffer(size=7)

# 테스트용 파라미터 (batch 처리 전용)
state_manager = StableStateManager(
    min_duration=0.0,  # 시간 조건 제거
    min_confidence=0.5,  # 완화
)


def run_batch_ocr():
    img_dir: Path = PATHS.TEST_BANPICK_STATUS_DIR

    if not img_dir.exists():
        print("❌ 테스트 이미지 폴더 없음:", img_dir)
        return

    img_files = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )

    print(f"📂 OCR 대상 이미지 수: {len(img_files)}")

    results = []

    for idx, img_path in enumerate(img_files):
        try:
            img = Image.open(img_path)

            # OCR
            text = extract_text(img)

            # Pipeline
            norm = normalizer.normalize(text)
            state = classifier.classify(norm)

            buffer.push(state)
            candidate = buffer.get_majority()
            confidence = buffer.get_confidence()

            stable_state = state_manager.update(candidate, confidence)

            record = {
                "file": img_path.name,
                "ocr": text,
                "normalized": norm,
                "classified": state,
                "buffer_majority": candidate,
                "confidence": round(confidence, 2),
                "stable_state": stable_state,
            }

            results.append(record)

            # 로그 출력
            print(f"🖼 {img_path.name}")
            # print(f" OCR        → {text}")
            print(f" Normalize   → {norm}")
            # print(f" Classify    → {state}")
            # print(f" Buffer      → {candidate} ({confidence:.2f})")
            print(f" StableState → {stable_state}")
            print("-" * 60)

        except Exception as e:
            print(f"❌ 실패: {img_path.name} | {e}")

    return results


if __name__ == "__main__":
    run_batch_ocr()

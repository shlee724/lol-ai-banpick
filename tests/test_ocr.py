from pathlib import Path
from PIL import Image
from core.ocr_engine import extract_text
from config.path import PATHS


def run_batch_ocr():
    img_dir: Path = PATHS.TEST_BANPICK_STATUS_DIR

    if not img_dir.exists():
        print("❌ 테스트 이미지 폴더 없음:", img_dir)
        return

    img_files = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ])

    print(f"📂 OCR 대상 이미지 수: {len(img_files)}")

    results = []

    for img_path in img_files:
        try:
            img = Image.open(img_path)
            text = extract_text(img)

            result = {
                "file": img_path.name,
                "text": text
            }

            results.append(result)

            print(f"🖼 {img_path.name}")
            print(f"   OCR → {text}")
            print("-" * 50)

        except Exception as e:
            print(f"❌ 실패: {img_path.name} | {e}")

    return results


if __name__ == "__main__":
    run_batch_ocr()

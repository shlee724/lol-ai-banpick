from pathlib import Path
from PIL import Image

from pipeline.ban_detector import detect_ban_strip_variance
from config.path import PATHS


def run_ban_detector_batch(std_threshold: float = 18.0):
    img_dir: Path = PATHS["TEST_BANNED_SLOTS"]

    if not img_dir.exists():
        print("❌ banned_slots 폴더 없음:", img_dir)
        return

    img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() == ".png"])

    print(f"📂 banned_slots 이미지 수: {len(img_files)}")
    print(f"🧪 std_threshold = {std_threshold}\n")

    filled_count = 0
    results = []

    for p in img_files:
        try:
            img = Image.open(p)
            res = detect_ban_strip_variance(img, std_threshold=std_threshold)

            filled_count += 1 if res.is_filled else 0
            results.append((p.name, res.std, res.is_filled))

            tag = "FILLED" if res.is_filled else "EMPTY"
            print(f"🖼 {p.name:<40}  std={res.std:>6.2f}  -> {tag}")

        except Exception as e:
            print(f"❌ 실패: {p.name} | {e}")

    print("\n--- 요약 ---")
    print(f"FILLED: {filled_count} / {len(img_files)}")
    print(f"EMPTY : {len(img_files) - filled_count} / {len(img_files)}")

    return results


if __name__ == "__main__":
    # threshold만 바꿔가면서 튜닝
    run_ban_detector_batch(std_threshold=18.0)

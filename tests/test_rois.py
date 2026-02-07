from pathlib import Path
from PIL import Image
from config.roi import ROI
from config.path import PATHS
from core.roi_manager import crop_roi_relative_xy


def run_batch_roi_cut():
    img_dir: Path = PATHS["GEN_TEST_LOL_CLIENT_CAPTURE"]
    save_dir: Path = PATHS["TEST_BANNED_SLOTS"]

    if not img_dir.exists():
        print("❌ 테스트 이미지 폴더 없음:", img_dir)
        return

    save_dir.mkdir(parents=True, exist_ok=True)  # ✅ 저장 폴더 보장

    img_files = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ])

    print(f"📂 대상 이미지 수: {len(img_files)}")

    results = []

    for img_path in img_files:
        try:
            img = Image.open(img_path)

            rect = (0, 0, 1600, 900)
            my_banned = crop_roi_relative_xy(img, rect, ROI["banned_champions_area_my_team"])
            enemy_banned = crop_roi_relative_xy(img, rect, ROI["banned_champions_area_enemy_team"])

            total_banned = merge_images_horizontal(my_banned, enemy_banned)

            save_path = save_dir / f"{img_path.stem}_banned_slots.png"
            total_banned.save(save_path)

            # 원하면 결과 기록
            results.append(save_path)

        except Exception as e:
            print(f"❌ 실패: {img_path.name} | {e}")

    return results


def merge_images_horizontal(img1: Image.Image, img2: Image.Image, bg_color=(255, 255, 255)) -> Image.Image:
    new_width = img1.width + img2.width
    new_height = max(img1.height, img2.height)

    new_img = Image.new("RGB", (new_width, new_height), bg_color)
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    return new_img


if __name__ == "__main__":
    run_batch_roi_cut()

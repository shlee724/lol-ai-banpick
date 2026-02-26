from pathlib import Path
from typing import List

from PIL import Image

from config.path import PATHS
from config.roi import ROI
from core.roi_manager import crop_roi_relative_xy


def run_batch_banned_slots_roi_cut():
    img_dir: Path = PATHS.TEST_LOL_CLIENT_DIR
    save_dir: Path = PATHS.TEST_BANNED_SLOTS_DIR

    if not img_dir.exists():
        print("❌ 테스트 이미지 폴더 없음:", img_dir)
        return

    save_dir.mkdir(parents=True, exist_ok=True)  # ✅ 저장 폴더 보장

    img_files = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )

    print(f"📂 대상 이미지 수: {len(img_files)}")

    results = []

    for img_path in img_files:
        try:
            img = Image.open(img_path)

            window_size = (1600, 900)
            my_banned = crop_roi_relative_xy(img, window_size, ROI.BANNED_CHAMPIONS_MY_TEAM)
            enemy_banned = crop_roi_relative_xy(img, window_size, ROI.BANNED_CHAMPIONS_ENEMY_TEAM)

            total_banned = merge_images_horizontal(my_banned, enemy_banned)

            save_path = save_dir / f"{img_path.stem}_banned_slots.png"
            total_banned.save(save_path)

            # 원하면 결과 기록
            results.append(save_path)

        except Exception as e:
            print(f"❌ 실패: {img_path.name} | {e}")

    return results


def run_batch_picked_champs_roi_cut():
    img_dir: Path = PATHS.TEST_LOL_CLIENT_DIR
    save_dir: Path = PATHS.TEST_PICKED_CHAMPS_DIR

    if not img_dir.exists():
        print("❌ 테스트 이미지 폴더 없음:", img_dir)
        return

    save_dir.mkdir(parents=True, exist_ok=True)  # ✅ 저장 폴더 보장

    img_files = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )

    print(f"📂 대상 이미지 수: {len(img_files)}")

    results = []

    for img_path in img_files:
        try:
            img = Image.open(img_path)

            window_size = (1600, 900)
            my_picked = crop_roi_relative_xy(img, window_size, ROI.PICKED_CHAMPIONS_MY_TEAM)
            enemy_picked = crop_roi_relative_xy(img, window_size, ROI.PICKED_CHAMPIONS_ENEMY_TEAM)

            total_picked = merge_images_horizontal(my_picked, enemy_picked)

            save_path = save_dir / f"{img_path.stem}_picked_champs.png"
            total_picked.save(save_path)

            # 원하면 결과 기록
            results.append(save_path)

        except Exception as e:
            print(f"❌ 실패: {img_path.name} | {e}")

    return results


def run_batch_banpick_timer_bar_roi_cut():
    img_dir: Path = PATHS.TEST_LOL_CLIENT_DIR
    save_dir: Path = PATHS.TEST_BANPICK_TIMER_DIR

    if not img_dir.exists():
        print("❌ 테스트 이미지 폴더 없음:", img_dir)
        return

    save_dir.mkdir(parents=True, exist_ok=True)  # ✅ 저장 폴더 보장

    img_files = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )

    print(f"📂 대상 이미지 수: {len(img_files)}")

    results = []

    for img_path in img_files:
        try:
            img = Image.open(img_path)

            window_size = (1600, 900)
            banpick_timer = crop_roi_relative_xy(img, window_size, ROI.BANPICK_TIMER_BAR)

            save_path = save_dir / f"{img_path.stem}_timer_bar.png"
            banpick_timer.save(save_path)

            # 원하면 결과 기록
            results.append(save_path)

        except Exception as e:
            print(f"❌ 실패: {img_path.name} | {e}")

    return results


def run_batch_picked_champs_text_only_roi_cut():
    img_dir: Path = PATHS.TEST_LOL_CLIENT_DIR
    save_dir: Path = PATHS.TEST_PICKED_CHAMPS_TEXT_ONLY_DIR

    if not img_dir.exists():
        print("❌ 테스트 이미지 폴더 없음:", img_dir)
        return

    save_dir.mkdir(parents=True, exist_ok=True)  # ✅ 저장 폴더 보장

    img_files = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )

    print(f"📂 대상 이미지 수: {len(img_files)}")

    results = []

    for img_path in img_files:
        try:
            img = Image.open(img_path)

            window_size = (1600, 900)
            my_picked_1 = crop_roi_relative_xy(img, window_size, ROI.MY_TEAM_PICK1)
            my_picked_2 = crop_roi_relative_xy(img, window_size, ROI.MY_TEAM_PICK2)
            my_picked_3 = crop_roi_relative_xy(img, window_size, ROI.MY_TEAM_PICK3)
            my_picked_4 = crop_roi_relative_xy(img, window_size, ROI.MY_TEAM_PICK4)
            my_picked_5 = crop_roi_relative_xy(img, window_size, ROI.MY_TEAM_PICK5)
            enemy_picked_1 = crop_roi_relative_xy(img, window_size, ROI.ENEMY_TEAM_PICK1)
            enemy_picked_2 = crop_roi_relative_xy(img, window_size, ROI.ENEMY_TEAM_PICK2)
            enemy_picked_3 = crop_roi_relative_xy(img, window_size, ROI.ENEMY_TEAM_PICK3)
            enemy_picked_4 = crop_roi_relative_xy(img, window_size, ROI.ENEMY_TEAM_PICK4)
            enemy_picked_5 = crop_roi_relative_xy(img, window_size, ROI.ENEMY_TEAM_PICK5)
            my_pos_1 = crop_roi_relative_xy(img, window_size, ROI.MY_TEAM_POS1)
            my_pos_2 = crop_roi_relative_xy(img, window_size, ROI.MY_TEAM_POS2)
            my_pos_3 = crop_roi_relative_xy(img, window_size, ROI.MY_TEAM_POS3)
            my_pos_4 = crop_roi_relative_xy(img, window_size, ROI.MY_TEAM_POS4)
            my_pos_5 = crop_roi_relative_xy(img, window_size, ROI.MY_TEAM_POS5)

            my_list = [
                my_pos_1,
                my_picked_1,
                my_pos_2,
                my_picked_2,
                my_pos_3,
                my_picked_3,
                my_pos_4,
                my_picked_4,
                my_pos_5,
                my_picked_5,
            ]
            enemy_list = [
                enemy_picked_1,
                enemy_picked_2,
                enemy_picked_3,
                enemy_picked_4,
                enemy_picked_5,
            ]

            my_picked_merge = merge_images_vertical(images=my_list)
            enemy_picked_merge = merge_images_vertical(images=enemy_list)

            total_picked = merge_images_horizontal(my_picked_merge, enemy_picked_merge)

            save_path = save_dir / f"{img_path.stem}_picked_champs.png"
            total_picked.save(save_path)

            # 원하면 결과 기록
            results.append(save_path)

        except Exception as e:
            print(f"❌ 실패: {img_path.name} | {e}")

    return results


def merge_images_horizontal(
    img1: Image.Image, img2: Image.Image, bg_color=(255, 255, 255)
) -> Image.Image:
    new_width = img1.width + img2.width
    new_height = max(img1.height, img2.height)

    new_img = Image.new("RGB", (new_width, new_height), bg_color)
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    return new_img


def merge_images_vertical(images: List[Image.Image], bg_color=(255, 255, 255)) -> Image.Image:
    """
    여러 이미지를 세로로 이어 붙인다.
    - 너비는 가장 넓은 이미지 기준
    - 빈 공간은 bg_color로 채움
    """

    if not images:
        raise ValueError("이미지 리스트가 비어 있음")

    new_width = max(img.width for img in images)
    new_height = sum(img.height for img in images)

    new_img = Image.new("RGB", (new_width, new_height), bg_color)

    y_offset = 0
    for img in images:
        new_img.paste(img, (0, y_offset))
        y_offset += img.height

    return new_img


if __name__ == "__main__":
    run_batch_banpick_timer_bar_roi_cut()

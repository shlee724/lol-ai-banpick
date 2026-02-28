from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from PIL import Image

from config.roi import ROI
from core.roi_manager import crop_roi_relative_xy

@dataclass(frozen=True)
class Rois:
    status_img: Image.Image
    bans_my_img: Image.Image
    bans_enemy_img: Image.Image
    picks_my_img: Image.Image
    picks_enemy_img: Image.Image
    picks_merged_img: Image.Image
    timer_bar_img: Image.Image
    timer_digits_img: Image.Image

def merge_images_horizontal(img_left: Image.Image, img_right: Image.Image, bg_color=(255,255,255)) -> Image.Image:
    new_width = img_left.width + img_right.width
    new_height = max(img_left.height, img_right.height)
    merged = Image.new("RGB", (new_width, new_height), bg_color)
    merged.paste(img_left, (0, 0))
    merged.paste(img_right, (img_left.width, 0))
    return merged

def extract_rois(frame_img: Image.Image, window_size: Tuple[int,int]) -> Rois:
    status_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANPICK_STATUS_TEXT)

    bans_my_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANNED_CHAMPIONS_MY_TEAM)
    bans_enemy_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANNED_CHAMPIONS_ENEMY_TEAM)

    picks_my_img = crop_roi_relative_xy(frame_img, window_size, ROI.PICKED_CHAMPIONS_MY_TEAM)
    picks_enemy_img = crop_roi_relative_xy(frame_img, window_size, ROI.PICKED_CHAMPIONS_ENEMY_TEAM)
    picks_merged_img = merge_images_horizontal(picks_my_img, picks_enemy_img)

    timer_bar_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANPICK_TIMER_BAR)
    timer_digits_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANPICK_TIMER_DIGITS)

    return Rois(
        status_img=status_img,
        bans_my_img=bans_my_img,
        bans_enemy_img=bans_enemy_img,
        picks_my_img=picks_my_img,
        picks_enemy_img=picks_enemy_img,
        picks_merged_img=picks_merged_img,
        timer_bar_img=timer_bar_img,
        timer_digits_img=timer_digits_img,
    )
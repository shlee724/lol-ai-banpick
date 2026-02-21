# config/path.py
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # TEST_PROJECT

@dataclass(frozen=True)
class Paths:
    CAPTURE_DIR: Path = PROJECT_ROOT / "captured_images"
    TEST_IMAGES_DIR: Path = CAPTURE_DIR / "test_images"

    LOL_CLIENT_CAPTURE_PNG: Path = CAPTURE_DIR / "lol_client_capture.png"
    BANPICK_STATUS_TEXT_CAPTURE_PNG: Path = CAPTURE_DIR / "banpick_status_text_capture.png"

    TEST_LOL_CLIENT_DIR: Path = TEST_IMAGES_DIR / "lol_client"
    TEST_BANPICK_STATUS_DIR: Path = TEST_IMAGES_DIR / "banpick_status"
    TEST_BANPICK_TIMER_DIR: Path = TEST_IMAGES_DIR / "banpick_timer"

    TEST_BANNED_SLOTS_DIR: Path = TEST_IMAGES_DIR / "banned_slots"
    TEST_PICKED_CHAMPS_DIR: Path = TEST_IMAGES_DIR / "picked_champs"
    TEST_PICKED_CHAMPS_TEXT_ONLY_DIR: Path = TEST_IMAGES_DIR / "picked_champs_text_only"

PATHS = Paths()

# config/path.py
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # TEST_PROJECT

@dataclass(frozen=True)
class Paths:
    CAPTURE_DIR: Path = PROJECT_ROOT / "captured_images"
    TEST_IMAGES_DIR: Path = CAPTURE_DIR / "test_images"

    LOL_CLIENT_CAPTURE: Path = CAPTURE_DIR / "lol_client_capture.png"
    BANPICK_STATUS_TEXT_CAPTURE: Path = CAPTURE_DIR / "banpick_status_text_capture.png"

    GEN_TEST_LOL_CLIENT_CAPTURE: Path = TEST_IMAGES_DIR / "lol_client"
    GEN_TEST_BANPICK_STATUS_CAPTURE: Path = TEST_IMAGES_DIR / "banpick_status"

    TEST_BANNED_SLOTS: Path = TEST_IMAGES_DIR / "banned_slots"
    TEST_PICKED_CHAMPS: Path = TEST_IMAGES_DIR / "picked_champs"

PATHS = Paths()

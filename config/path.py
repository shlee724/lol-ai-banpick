# config/path.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # TEST_PROJECT 기준

PATHS = {
    "CAPTURE_DIR": BASE_DIR / "captured_images",
    "LOL_CLIENT_CAPTURE": BASE_DIR / "captured_images" / "lol_client_capture.png",
    "BANPICK_STATUS_TEXT_CAPTURE": BASE_DIR / "captured_images" / "banpick_status_text_capture.png",
    "GEN_TEST_LOL_CLIENT_CAPTURE": BASE_DIR / "captured_images" / "test_images" / "lol_client",
    "GEN_TEST_BANPICK_STATUS_CAPTURE": BASE_DIR / "captured_images" / "test_images" / "banpick_status",
    "TEST_BANNED_SLOTS": BASE_DIR / "captured_images" / "test_images" / "banned_slots",
    "TEST_PICKED_CHAMPS": BASE_DIR / "captured_images" / "test_images" / "picked_champs",
}

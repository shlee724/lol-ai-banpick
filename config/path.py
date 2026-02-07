# config/path.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # TEST_PROJECT 기준

PATHS = {
    "CAPTURE_DIR": BASE_DIR / "captured_images",
    "LOL_CLIENT_CAPTURE": BASE_DIR / "captured_images" / "lol_client_capture.png",
    "BANPICK_STATUS_TEXT_CAPTURE": BASE_DIR / "captured_images" / "banpick_status_text_capture.png",
}

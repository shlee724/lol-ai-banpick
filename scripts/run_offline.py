from __future__ import annotations

# ======================
# Standard library
# ======================
import argparse
import time
from pathlib import Path

# ======================
# Third-party
# ======================
from PIL import Image

# ======================
# Local modules
# ======================
from config.path import PATHS

from app.settings import Settings
from app.rois import extract_rois
from app.loop import run_streaming   # 🔥 여기서 재사용

from core.ocr_engine import extract_text
from core.lol_pick_coach import get_client, lol_mid_pick_coach_stream
from core.lol_playplan_coach import get_playplan_coach_client, lol_playplan_stream

from pipeline.buffer import StateBuffer
from pipeline.classifier import StateClassifier
from pipeline.normalizer import TextNormalizer
from pipeline.pick_stage_detector import detect_pick_kind_from_banned_strips
from pipeline.prepare_phase_detector import is_dual_timer_effective
from pipeline.state_manager import StableStateManager


# ======================
# Helpers
# ======================
def list_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths, key=lambda p: p.name)


def open_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ======================
# Main
# ======================
def main() -> None:
    defaults = Settings()

    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", required=True)
    parser.add_argument("--sleep", type=float, default=defaults.sleep_sec)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no_api", action="store_true")

    parser.add_argument("--pick_std", type=float, default=defaults.pick_std_threshold)
    parser.add_argument("--dual_conf", type=float, default=defaults.dual_conf_threshold)
    parser.add_argument("--model", type=str, default=defaults.gemini_model)
    parser.add_argument("--state_buf", type=int, default=defaults.state_buf_size)
    parser.add_argument("--dual_buf", type=int, default=defaults.dual_buf_size)

    args = parser.parse_args()

    settings = Settings(
        sleep_sec=args.sleep,
        state_buf_size=args.state_buf,
        dual_buf_size=args.dual_buf,
        pick_std_threshold=args.pick_std,
        dual_conf_threshold=args.dual_conf,
        gemini_model=args.model,
        debug_save=defaults.debug_save,
        window_title=defaults.window_title,
    )

    test_dir = PATHS.TEST_LOL_CLIENT_DIR / args.testset
    if not test_dir.exists():
        raise FileNotFoundError(f"테스트셋 폴더 없음: {test_dir}")

    img_paths = list_images(test_dir)
    if not img_paths:
        raise FileNotFoundError(f"이미지 없음: {test_dir}")

    # ----------------------
    # Init (main.py와 동일)
    # ----------------------
    normalizer = TextNormalizer()
    classifier = StateClassifier()

    state_buf = StateBuffer(size=settings.state_buf_size)
    dual_buf = StateBuffer(size=settings.dual_buf_size)

    state_manager = StableStateManager(min_duration=1.0, min_confidence=0.7)

    pick_coach_client = None
    playplan_coach_client = None
    if not args.no_api:
        pick_coach_client = get_client()
        playplan_coach_client = get_playplan_coach_client()

    pick_real_executed = False

    print(f"📁 OFFLINE testset: {test_dir}")
    print(f"🖼 frames: {len(img_paths)} | no_api={args.no_api} | sleep={settings.sleep_sec}")
    print("====================================")

    processed = 0

    # ----------------------
    # Frame iteration
    # ----------------------
    for idx, frame_path in enumerate(img_paths, start=1):
        if args.limit and processed >= args.limit:
            break

        frame_img = open_rgb(frame_path)
        window_size = frame_img.size

        rois = extract_rois(frame_img, window_size)

        status_text_raw = extract_text(rois.status_img)
        status_text_norm = normalizer.normalize(status_text_raw)
        raw_state = classifier.classify(status_text_norm)

        state_buf.push(raw_state)
        major_state = state_buf.get_majority()
        major_conf = state_buf.get_confidence()

        stable_state = state_manager.update(major_state, major_conf)

        print(f"\n#[{idx:04d}] {frame_path.name}")
        print(
            f" OCR='{status_text_raw}' | norm='{status_text_norm}'"
            f" | raw={raw_state} | major={major_state}({major_conf:.2f}) | stable={stable_state}"
        )

        # ----------------------
        # State actions
        # ----------------------
        if stable_state == "PICK":
            if raw_state == "BAN" or pick_real_executed:
                processed += 1
                time.sleep(settings.sleep_sec)
                continue

            pick_res = detect_pick_kind_from_banned_strips(
                rois.bans_my_img,
                rois.bans_enemy_img,
                std_threshold=settings.pick_std_threshold,
            )
            print(f"[PICK] 판정: kind={pick_res.kind} std={pick_res.std:.2f}")

            if pick_res.kind == "PICK_REAL":
                if args.no_api:
                    print("[PICK] (no_api) 호출 생략")
                else:
                    try:
                        run_streaming(
                            "PICK_COACH",
                            lol_mid_pick_coach_stream(
                                rois.picks_merged_img,
                                client=pick_coach_client,
                                model=settings.gemini_model,
                            ),
                        )
                    except Exception as e:
                        print("[ERR] Gemini 호출 실패:", repr(e))

                pick_real_executed = True
                processed += 1
                time.sleep(settings.sleep_sec)
                continue

        elif stable_state == "PREPARE":
            dual_now = is_dual_timer_effective(
                timer_bar_img=rois.timer_bar_img,
                timer_digits_img=rois.timer_digits_img,
            )

            dual_buf.push(dual_now)
            dual_stable = dual_buf.get_majority()
            dual_conf = dual_buf.get_confidence()

            print(f"[PREPARE] DualEffective: now={dual_now} stable={dual_stable} ({dual_conf:.2f})")

            if dual_stable and dual_conf >= settings.dual_conf_threshold:
                print("[PREPARE] 모든 챔피언 픽 완료 (stable)")

                if args.no_api:
                    print("[PREPARE] (no_api) 호출 생략")
                else:
                    run_streaming(
                        "PLAYPLAN_COACH",
                        lol_playplan_stream(
                            rois.picks_merged_img,
                            client=playplan_coach_client,
                            model=settings.gemini_model,
                        ),
                    )
                break

        else:
            dual_buf.reset()

        processed += 1
        time.sleep(settings.sleep_sec)

    print("\n====================================")
    print(f"✅ OFFLINE DONE. processed={processed} / total={len(img_paths)}")


if __name__ == "__main__":
    main()
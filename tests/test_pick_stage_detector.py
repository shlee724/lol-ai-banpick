# tests/test_pick_stage_detector.py

from pathlib import Path
from PIL import Image
import time
import re

from config.path import PATHS
from config.roi import ROI
from core.roi_manager import crop_roi_relative_xy
from core.ocr_engine import extract_text

from pipeline.normalizer import TextNormalizer
from pipeline.classifier import StateClassifier
from pipeline.buffer import StateBuffer
from pipeline.state_manager import StableStateManager
from pipeline.pick_stage_detector import detect_pick_kind_from_banned_strips


TS_PATTERN = re.compile(r".*_(\d{10,})\.(png|jpg|jpeg)$", re.IGNORECASE)


def _extract_ts_ms(filename: str) -> int | None:
    """
    lol_client_1770452190299.png 같은 파일명에서 timestamp(ms) 추출.
    없으면 None.
    """
    m = TS_PATTERN.match(filename)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def run_offline_like_main(
    *,
    std_threshold: float = 25.0,
    buffer_size: int = 7,
    min_duration: float = 1.0,
    min_confidence: float = 0.7,
    simulate_timing: bool = True,
    max_sleep_sec: float = 0.5,
):
    """
    main.py의 흐름을 그대로 'lol_client' 폴더 이미지 순회로 재현.
    - simulate_timing=True면 파일명 timestamp 차이로 sleep을 흉내내서
      StableStateManager(min_duration)이 실제처럼 동작하게 함.
    """
    img_dir: Path = PATHS.TEST_LOL_CLIENT_DIR

    if not img_dir.exists():
        print("❌ lol_client 테스트 폴더 없음:", img_dir)
        return

    img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    print(f"📂 lol_client 이미지 수: {len(img_files)}")
    print(
        f"설정: buffer={buffer_size}, min_duration={min_duration}, min_conf={min_confidence}, "
        f"pick_std_th={std_threshold}, simulate_timing={simulate_timing}\n"
    )

    # main.py와 동일 구성 :contentReference[oaicite:1]{index=1}
    normalizer = TextNormalizer()
    classifier = StateClassifier()
    buffer = StateBuffer(size=buffer_size)
    state_manager = StableStateManager(min_duration=min_duration, min_confidence=min_confidence)

    prev_ts_ms = None

    for idx, img_path in enumerate(img_files):
        try:
            img = Image.open(img_path)

            # crop_roi_relative_xy가 rect의 w/h를 쓸 가능성이 높아서
            # 테스트에서는 (0,0, img.width, img.height)로 넣어줌
            window_size = (img.width, img.height)

            # main.py와 동일 ROI crop :contentReference[oaicite:2]{index=2}
            status_img = crop_roi_relative_xy(img, window_size, ROI["banpick_status_text"])
            my_banned = crop_roi_relative_xy(img, window_size, ROI["banned_champions_area_my_team"])
            enemy_banned = crop_roi_relative_xy(img, window_size, ROI["banned_champions_area_enemy_team"])

            # OCR
            text = extract_text(status_img)

            # Pipeline
            norm = normalizer.normalize(text)
            state = classifier.classify(norm)

            buffer.push(state)
            candidate = buffer.get_majority()
            confidence = buffer.get_confidence()

            stable_state = state_manager.update(candidate, confidence)

            # 로그
            print(f"#{idx:04d} 🖼 {img_path.name}")
            print(f" OCR        → {text}")
            print(f" Normalize   → {norm}")
            print(f" Classify    → {state}")
            print(f" Buffer      → {candidate} ({confidence:.2f})")
            print(f" StableState → {stable_state}")

            # PICK일 때 REAL/FAKE 판정(main과 동일) :contentReference[oaicite:3]{index=3}
            if stable_state == "PICK":
                pick_res = detect_pick_kind_from_banned_strips(
                    my_banned, enemy_banned, std_threshold=std_threshold
                )
                print(f" PICK 판정  → {pick_res.kind} (std={pick_res.std:.2f})")

            print("-" * 60)

            # timing 시뮬레이션: 파일명 타임스탬프로 sleep
            if simulate_timing:
                ts_ms = _extract_ts_ms(img_path.name)
                if ts_ms is not None and prev_ts_ms is not None:
                    dt = max(0.0, (ts_ms - prev_ts_ms) / 1000.0)
                    time.sleep(min(dt, max_sleep_sec))
                elif ts_ms is None:
                    # timestamp 없는 파일명일 경우 main처럼 0.3초 흉내
                    time.sleep(0.3)

                if ts_ms is not None:
                    prev_ts_ms = ts_ms

        except Exception as e:
            print(f"❌ 실패: {img_path.name} | {e}")
            print("-" * 60)


if __name__ == "__main__":
    # main.py에 들어간 파라미터를 그대로 기본값으로 맞춰둠 :contentReference[oaicite:4]{index=4}
    run_offline_like_main(
        std_threshold=25.0,
        buffer_size=7,
        min_duration=1.0,
        min_confidence=0.7,
        simulate_timing=True,
        max_sleep_sec=0.5,
    )

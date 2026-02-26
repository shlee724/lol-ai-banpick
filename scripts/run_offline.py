from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Optional, Tuple

from PIL import Image

from app.settings import AppSettings
from app.wiring import build_deps
from app.run import run_loop_with_provider
from config.path import PATHS


def list_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths, key=lambda p: p.name)


def open_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def make_folder_provider(
    img_paths: list[Path],
    *,
    limit: int = 0,
) -> Tuple[callable, callable]:
    it: Iterator[Path] = iter(img_paths)
    i = {"n": 0}
    last = {"p": None}

    def provider() -> Optional[Tuple[Image.Image, Tuple[int, int]]]:
        if limit and i["n"] >= limit:
            return None
        try:
            p = next(it)
        except StopIteration:
            return None

        i["n"] += 1
        last["p"] = p

        frame = open_rgb(p)
        w, h = frame.size
        return frame, (w, h), p.name

    def print_frame_header() -> None:
        p = last["p"]
        if p is not None:
            print(f"\n#[{i['n']:04d}] {p.name}")

    return provider, print_frame_header


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", required=True, help="lol_client 하위 테스트셋 폴더명 (예: test_1)")
    parser.add_argument("--sleep", type=float, default=None, help="프레임 간 sleep (기본=AppSettings)")
    parser.add_argument("--limit", type=int, default=0, help="최대 처리 프레임 수 (0=무제한)")
    args = parser.parse_args()

    test_dir = PATHS.TEST_LOL_CLIENT_DIR / args.testset
    if not test_dir.exists():
        raise FileNotFoundError(f"테스트셋 폴더 없음: {test_dir}")

    img_paths = list_images(test_dir)
    if not img_paths:
        raise FileNotFoundError(f"이미지 없음: {test_dir}")

    settings = AppSettings()
    deps = build_deps(settings)

    print(f"📁 OFFLINE testset: {test_dir}")
    print(f"🖼 frames: {len(img_paths)} | sleep={args.sleep} | limit={args.limit}")
    print("====================================")

    provider, hook = make_folder_provider(img_paths, limit=args.limit)

    # 프레임 헤더 출력이 필요하면, 아래처럼 provider를 감싸면 됨
    def provider_with_log():
        out = provider()
        if out is not None:
            hook()
        return out

    run_loop_with_provider(settings, deps, provider_with_log, sleep_sec=args.sleep)

    print("\n====================================")
    print("✅ OFFLINE DONE.")


if __name__ == "__main__":
    main()
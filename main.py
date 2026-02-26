from __future__ import annotations

from app.settings import AppSettings
from app.wiring import build_deps
from app.run import run_loop


def main() -> None:
    settings = AppSettings()
    deps = build_deps(settings)
    run_loop(settings=settings, deps=deps)


if __name__ == "__main__":
    main()
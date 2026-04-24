from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_bootstrap()

from rl_isaac.runners.train import main


if __name__ == "__main__":
    raise SystemExit(main())

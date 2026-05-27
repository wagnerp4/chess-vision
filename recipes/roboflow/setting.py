from __future__ import annotations

import os
import sys
from pathlib import Path

from src.training.paths import repo_root


def bootstrap_repo() -> Path:
    root = repo_root()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root

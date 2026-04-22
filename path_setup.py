"""Insert project root on sys.path so `theory` and `publication` packages resolve."""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Repository root (folder containing theory/ and publication/). Exposed for stable path defaults.
REPO_ROOT = _root

"""
Single root for generated files (figures, CSV): ``<repo>/heat_output/``.

Subfolders are fixed so all scripts default into the same tree;
override with each tool's ``--out-dir``.
"""
from __future__ import annotations

from pathlib import Path

import path_setup

REPO_ROOT = path_setup.REPO_ROOT
HEAT_OUTPUT_ROOT = REPO_ROOT / "heat_output"

JWST_EARLY = HEAT_OUTPUT_ROOT / "jwst_early_galaxies"
SPARC_PUBLICATION = HEAT_OUTPUT_ROOT / "sparc_publication"


def ensure_dir(path: Path | str) -> Path:
    """Create directory if needed; return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


__all__ = [
    "JWST_EARLY",
    "REPO_ROOT",
    "HEAT_OUTPUT_ROOT",
    "SPARC_PUBLICATION",
    "ensure_dir",
]

"""SPARC Rotmod file helpers."""
from __future__ import annotations

import re


def extract_distance(file_path: str) -> float | None:
    """
    Read '# Distance = X.XX Mpc' from a SPARC .dat header.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("# Distance ="):
                    match = re.search(r"(\d+\.?\d*)", line)
                    if match:
                        return float(match.group(1))
        return None
    except OSError as e:
        print(f"Error reading {file_path}: {e}")
        return None

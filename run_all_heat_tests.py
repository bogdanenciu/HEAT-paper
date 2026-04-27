"""Run HEAT letter figure-generation scripts.

Usage:  python run_all_heat_tests.py list        # show available keys
        python run_all_heat_tests.py jwst --agg   # JWST analysis (non-interactive)
        python run_all_heat_tests.py all --agg    # everything

See HEAT_THEORY.md for details.
"""
from __future__ import annotations

import argparse
import importlib
import os

import path_setup  # noqa: F401  # project root on sys.path


TESTS = {
    "sparc-pub": "publication.sparc_publication_quality",
    "jwst": "publication.jwst_early_galaxies",
    "fig-norm": "publication.fig_normalization",
    "fig-mass": "publication.mass_selection_robustness",
}


def main():
    p = argparse.ArgumentParser(
        description="Run HEAT letter figure-generation scripts. See HEAT_THEORY.md."
    )
    p.add_argument(
        "which",
        nargs="?",
        default="list",
        choices=list(TESTS.keys()) + ["list", "all"],
        help="Which test to run, or 'all'",
    )
    p.add_argument("--agg", action="store_true", help="Use non-interactive Agg backend for matplotlib")
    args = p.parse_args()

    if args.agg:
        import matplotlib

        matplotlib.use("Agg")

    if args.which == "list":
        print("Available:", ", ".join(sorted(TESTS.keys())))
        print("Also: all")
        return

    keys = list(TESTS.keys()) if args.which == "all" else [args.which]
    failed: list[tuple[str, str]] = []
    for k in keys:
        try:
            mod = importlib.import_module(TESTS[k])
        except ImportError as e:
            print(f"\n--- Skipping {k} ({TESTS[k]}): import failed: {e} ---")
            failed.append((k, f"import: {e}"))
            continue
        fn = getattr(mod, "main", None)
        if fn is None:
            print(f"Skipping {k}: no main() in {TESTS[k]}")
            continue
        print(f"\n--- Running {k} ({TESTS[k]}) ---")
        try:
            if k.startswith("sparc"):
                fn(os.environ.get("SPARC_PATH", "Rotmod_LTG"))
            else:
                fn()
        except SystemExit as e:
            msg = f"SystemExit({e.code})"
            print(f"  [skipped] {msg}")
            failed.append((k, msg))
        except Exception as e:
            print(f"  [failed] {type(e).__name__}: {e}")
            failed.append((k, f"{type(e).__name__}: {e}"))
    if failed:
        print("\n--- Summary: the following tests did not complete ---")
        for k, reason in failed:
            print(f"  {k}: {reason}")


if __name__ == "__main__":
    main()

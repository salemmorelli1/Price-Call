#!/usr/bin/env python3
"""Build and verify the exact static bundle published by GitHub Pages."""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


DIRECTORIES = ("assets", "data", "report", "artifacts_dashboard")
PART10_FILES = ("signal_log.csv", "portfolio_state.json", "performance_report.json")


def prepare_site(root: Path, site: Path) -> None:
    site.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "index.html", site / "index.html")
    for name in DIRECTORIES:
        source = root / name
        if source.is_dir():
            shutil.copytree(
                source,
                site / name,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
    part10_site = site / "artifacts_part10_bot"
    part10_site.mkdir(parents=True, exist_ok=True)
    for name in PART10_FILES:
        source = root / "artifacts_part10_bot" / name
        if not source.is_file():
            raise FileNotFoundError(f"required Pages artifact missing: {source}")
        shutil.copy2(source, part10_site / name)
    favicon = root / "favicon.ico"
    if favicon.is_file():
        shutil.copy2(favicon, site / "favicon.ico")
    (site / ".nojekyll").touch()
    validate_site(site)


def validate_site(site: Path) -> None:
    html = (site / "index.html").read_text(encoding="utf-8")
    references = set(re.findall(r"(?:href=|src=|fetch\()['\"]([^'\"]+)", html))
    missing = []
    for reference in references:
        if reference.startswith(("#", "http://", "https://", "mailto:")):
            continue
        clean = reference.split("?", 1)[0].split("#", 1)[0]
        if clean and not (site / clean).is_file():
            missing.append(clean)
    if missing:
        raise FileNotFoundError("Pages bundle has unresolved local references: " + ", ".join(sorted(missing)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--site", default="_site")
    args = parser.parse_args()
    prepare_site(Path(args.root).resolve(), Path(args.site).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

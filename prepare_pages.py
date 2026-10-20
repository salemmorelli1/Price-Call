#!/usr/bin/env python3
"""Build and verify the exact static bundle published by GitHub Pages."""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from artifact_integrity import (
    REQUIRED_PUBLISHED_FILES,
    read_json_strict,
    validate_execution_lineage,
    validate_status_markers,
    verify_run_manifest,
    write_json_strict,
)
from sync_dashboard import build_snapshot, sync_html

DIRECTORIES = ("assets", "data", "report", "artifacts_dashboard")
PART10_FILES = (
    "signal_log.csv",
    "portfolio_state.json",
    "performance_report.json",
    "pipeline_status.json",
    "pipeline_run_date.txt",
)
PART9_FILES = ("backfill_status.json", "backfill_run_date.txt")


def _source_failures(root: Path) -> list[str]:
    return verify_run_manifest(root) + validate_status_markers(root) + validate_execution_lineage(root)


def _file_record(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def prepare_site(root: Path, site: Path, *, refresh_snapshot: bool = True) -> None:
    """Verify immutable inputs, then publish a complete bundle with an atomic rename."""
    root, site = root.resolve(), site.resolve()
    protected_names = set(DIRECTORIES) | {
        Path(name).parts[0] for name in REQUIRED_PUBLISHED_FILES if len(Path(name).parts) > 1
    }
    protected = [root / name for name in protected_names]
    if root.is_relative_to(site) or any(site.is_relative_to(path) for path in protected):
        raise ValueError("Pages output must not overlap source inputs")
    if site.exists():
        raise FileExistsError(f"Pages output already exists; use a fresh directory: {site}")
    failures = _source_failures(root)
    if failures:
        raise ValueError("Pages source verification failed: " + "; ".join(failures))
    source_manifest = (_file_record(root / "artifacts_manifest.json")
                       if (root / "artifacts_manifest.json").is_file() else None)
    site.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pricecall-pages-", dir=site.parent) as temporary:
        stage = Path(temporary) / "site"
        stage.mkdir()
        _copy_site(root, stage)
        if refresh_snapshot:
            write_json_strict(stage / "artifacts_dashboard/dashboard_snapshot.json", build_snapshot(root))
            sync_html(root, output_path=stage / "index.html")
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, capture_output=True, check=False,
        )
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            text=True, capture_output=True, check=False,
        )
        write_json_strict(stage / "publication.json", {
            "schema_version": "PRICECALL_PUBLICATION_V1",
            "builder_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "builder_source_modified": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
            "source_artifact_manifest": source_manifest,
            "builder_code_sha256": {
                name: _file_record(root / name)["sha256"] for name in (
                    "prepare_pages.py", "sync_dashboard.py", "evaluate_prospective_cohort.py",
                    "statistical_tests.py", "assets/snapshot_freshness.js",
                ) if (root / name).is_file()
            },
            "files": {path.relative_to(stage).as_posix(): _file_record(path)
                      for path in sorted(stage.rglob("*")) if path.is_file()},
        })
        validate_site(stage)
        # Detect a concurrent input change rather than publishing a mixed snapshot.
        current_manifest = (_file_record(root / "artifacts_manifest.json")
                            if (root / "artifacts_manifest.json").is_file() else None)
        if _source_failures(root) or current_manifest != source_manifest:
            raise ValueError("Pages sources changed during the build")
        stage.rename(site)


def _copy_site(root: Path, site: Path) -> None:
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
    part9_site = site / "artifacts_part9"
    part9_site.mkdir(parents=True, exist_ok=True)
    for name in PART9_FILES:
        source = root / "artifacts_part9" / name
        if not source.is_file():
            raise FileNotFoundError(f"required Pages artifact missing: {source}")
        shutil.copy2(source, part9_site / name)
    favicon = root / "favicon.ico"
    if favicon.is_file():
        shutil.copy2(favicon, site / "favicon.ico")
    (site / ".nojekyll").touch()


def validate_site(site: Path) -> None:
    html = (site / "index.html").read_text(encoding="utf-8")
    references = set(re.findall(r"(?:href=|src=|fetch\()['\"]([^'\"]+)", html))
    missing = []
    for reference in references:
        if reference.startswith(("#", "http://", "https://", "mailto:")):
            continue
        clean = reference.split("?", 1)[0].split("#", 1)[0]
        if clean and not (site / clean).resolve().is_relative_to(site.resolve()):
            raise ValueError(f"Pages reference escapes publication: {clean}")
        if clean and not (site / clean).is_file():
            missing.append(clean)
    if missing:
        raise FileNotFoundError("Pages bundle has unresolved local references: " + ", ".join(sorted(missing)))
    manifest = read_json_strict(site / "publication.json")
    if manifest.get("schema_version") != "PRICECALL_PUBLICATION_V1":
        raise ValueError("Unsupported Pages publication manifest")
    entries = manifest.get("files", {})
    actual = {path.relative_to(site).as_posix() for path in site.rglob("*")
              if path.is_file() and path != site / "publication.json"}
    if not entries or set(entries) != actual:
        raise ValueError("Pages publication file set differs from its manifest")
    for name, expected in entries.items():
        source = site / name
        path = source.resolve()
        if not path.is_relative_to(site.resolve()) or source.is_symlink():
            raise ValueError(f"Invalid Pages manifest path: {name}")
        if _file_record(path) != expected:
            raise ValueError(f"Pages publication hash differs: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--site", default="_site")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        validate_site(Path(args.site).resolve())
    else:
        prepare_site(Path(args.root).resolve(), Path(args.site).resolve())
    print("Pages publication verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

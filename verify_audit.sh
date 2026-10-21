#!/usr/bin/env bash
# The same verification entry point runs in Windows Git Bash and Linux CI.
set -Eeuo pipefail
trap 'printf "Stopped at line %s (exit %s).\n" "$LINENO" "$?" >&2' ERR

verification_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$verification_root"
git status --short
test -z "$(git status --porcelain)"

mkdir -p .venv
verification_parent="$(mktemp -d "$verification_root/.venv/audit-check.XXXXXX")"
verification_runtime="$verification_parent/runtime"

if python -c 'import sys; assert sys.version_info[:2] == (3, 12)' >/dev/null 2>&1; then
  python -m venv "$verification_runtime"
elif command -v py >/dev/null 2>&1; then
  py -3.12 -m venv "$verification_runtime"
else
  printf 'Python 3.12 is required. Install it, then rerun this script.\n' >&2
  exit 1
fi

if [ -f "$verification_runtime/Scripts/activate" ]; then
  source "$verification_runtime/Scripts/activate"
else
  source "$verification_runtime/bin/activate"
fi

export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PRICECALL_STRICT_DRIVE_ONLY=0
export PRICECALL_ROOT="$(python -c 'from pathlib import Path; print(Path.cwd())')"

python -c 'import sys; assert sys.version_info[:2] == (3, 12); assert sys.prefix != sys.base_prefix'
python -m pip install --no-cache-dir -r requirements-ci-lock.txt
python -m pip install --no-cache-dir duckdb==1.5.5 pyarrow==23.0.1
python -m pip check
python -m compileall -q ./*.py tests report
python -m ruff check --select E9,F63,F7,F82 .
python -m pytest -q tests/
GITHUB_REF_NAME=main python -m pytest -q tests/
python artifact_integrity.py --root .
python audit_governance.py --root .

verification_site="$verification_parent/site"
python prepare_pages.py --root . --site "$verification_site"
python prepare_pages.py --site "$verification_site" --verify

git diff --check
git diff --exit-code
test -z "$(git status --porcelain)"
git rev-parse HEAD
printf 'Verified preview: %s\n' "$verification_site"

#!/usr/bin/env bash
# Emberline verification: lint + full test suite + 60-second end-to-end smoke.
# Works under Git Bash on Windows and any POSIX shell.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

echo "== lint (pyflakes if present, else compileall) =="
if python -m pyflakes --version >/dev/null 2>&1; then
    python -m pyflakes emberline tests
else
    python -m compileall -q emberline tests
fi

echo "== unit tests =="
python -m pytest -q

echo "== end-to-end smoke (fast demo, physics fallback allowed) =="
python -m emberline.demo --scenario ridgeline --fast --kill-node N3 --wind-shift 35 \
    | tail -25

echo "VERIFY: ALL GREEN"

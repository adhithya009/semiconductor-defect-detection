#!/usr/bin/env bash
# Train every algorithm with the same budget, then print the comparison.
cd "$(dirname "$0")"
for script in algorithms/[0-9]*.py; do .venv/bin/python "$script" "$@" || echo "FAILED: $script"; done
.venv/bin/python compare.py

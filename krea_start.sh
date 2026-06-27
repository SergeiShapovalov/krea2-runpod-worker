#!/usr/bin/env bash
set -euo pipefail

python -u /prepare_models.py
exec /start.sh

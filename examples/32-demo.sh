#!/usr/bin/env bash
set -euo pipefail
python -m marginlab report
python -m marginlab audit
python -m marginlab controls >/dev/null

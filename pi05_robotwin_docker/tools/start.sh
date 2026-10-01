#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python3 tools/check_payload.py
docker compose version
docker compose up --build -d train
docker compose logs -f --tail 30 train

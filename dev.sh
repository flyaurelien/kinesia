#!/usr/bin/env bash
# Kinesia: the web app (UI + API) on http://127.0.0.1:4001.
# Processing runs on the remote GPU server; see cluster/config.env.example.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}/web-viewer"

if [ ! -e node_modules ]; then
  npm install
fi

exec env KINESIA_ROOT="${ROOT_DIR}" npm run dev -- --hostname 127.0.0.1 --port 4001

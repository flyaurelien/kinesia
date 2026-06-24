#!/usr/bin/env bash
# Kinesia dev launcher — single command to run the whole stack on port 4001.
# Usage: ./dev.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="${ROOT_DIR}/web-viewer"

cd "${WEB_DIR}"

# Ensure web dependencies are installed.
if [ ! -e node_modules ]; then
  npm install
fi

exec env \
  NEXT_PUBLIC_KINESIA_BACKEND_URL="" \
  NEXT_PUBLIC_KINESIA_BASIC_UI="0" \
  KINESIA_ALLOWED_ORIGINS="http://127.0.0.1:4001,http://localhost:4001" \
  SAM3D_MHR_MODE="${SAM3D_MHR_MODE:-native}" \
  `# SAM3 detector needs the per-op CPU fallback on Apple Silicon (MPS lacks` \
  `# aten::_assert_async); with it off, detection silently finds nobody.` \
  PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}" \
  npm run dev -- --hostname 127.0.0.1 --port 4001

#!/usr/bin/env bash
# Main-river recompute, remote environment: unpack the bundle zips found in BASE into BASE/root
# (code first, then the data zips) and build a pinned venv (exact interpreter patch, uv.lock,
# --inexact with explicit groups). Never plain `uv sync`.
#   bash setup.sh BASE [UV_CACHE]
set -euo pipefail
BASE=$(cd "$1" && pwd)
PY=3.12.13
export UV_CACHE_DIR="${2:-$BASE/uv-cache}" UV_PYTHON_INSTALL_DIR="$BASE/uv-python"
export UV_PROJECT_ENVIRONMENT="$BASE/root/.venv" UV_NO_CONFIG=1
mkdir -p "$BASE/root" "$BASE/logs"
cd "$BASE/root"
for z in code surfaces data receipts; do
  if [ -f "$BASE/$z.zip" ]; then python3 -m zipfile -e "$BASE/$z.zip" . ; fi
done
uv python install "$PY"
uv sync --frozen --inexact --no-default-groups --group ml --group hydro --python "$PY"
"$UV_PROJECT_ENVIRONMENT/bin/python" - "$PY" "$BASE/logs/startup.json" <<'PYEOF'
import json, platform, sys
from importlib import metadata
want, out = sys.argv[1], sys.argv[2]
got = platform.python_version()
pk = {p: metadata.version(p) for p in ("numpy", "scipy", "pandas", "geopandas", "shapely",
                                       "pyarrow", "rasterio", "exactextract")}
rec = {"python": got, "packages": pk, "machine": platform.machine(), "pin_ok": got == want}
json.dump(rec, open(out, "w"), indent=1)
print(json.dumps(rec))
sys.exit(0 if got == want else 3)
PYEOF

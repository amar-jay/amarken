#!/usr/bin/env bash
# Run on the GPU host after extracting the source bundle.
set -euo pipefail
AMARKEN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AMARKEN_ROOT"
AMARKEN_STORAGE="${AMARKEN_STORAGE:-$AMARKEN_ROOT/.cloud-state}"
mkdir -p "$AMARKEN_STORAGE"
AMARKEN_STORAGE="$(cd "$AMARKEN_STORAGE" && pwd)"
for name in .cache data runs outputs; do
  mkdir -p "$AMARKEN_STORAGE/$name"
  if [[ ! -e "$AMARKEN_ROOT/$name" && ! -L "$AMARKEN_ROOT/$name" ]]; then
    ln -s "$AMARKEN_STORAGE/$name" "$AMARKEN_ROOT/$name"
  elif [[ "$(readlink -f "$AMARKEN_ROOT/$name")" != "$AMARKEN_STORAGE/$name" ]]; then
    echo "Existing $name is outside AMARKEN_STORAGE; choose a fresh checkout or move it explicitly." >&2
    exit 1
  fi
done
export UV_CACHE_DIR="$AMARKEN_STORAGE/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$AMARKEN_STORAGE/.cache/python"
export MPLCONFIGDIR="$AMARKEN_STORAGE/.cache/matplotlib"
if [[ ! -x "$AMARKEN_ROOT/.cloud-bin/uv" ]]; then
  mkdir -p "$AMARKEN_ROOT/.cloud-bin"
  curl -LsSf https://astral.sh/uv/0.12.5/install.sh -o "$AMARKEN_ROOT/.cloud-bin/install-uv.sh"
  UV_INSTALL_DIR="$AMARKEN_ROOT/.cloud-bin" UV_NO_MODIFY_PATH=1 sh "$AMARKEN_ROOT/.cloud-bin/install-uv.sh"
fi
"$AMARKEN_ROOT/.cloud-bin/uv" sync --locked --no-dev --python 3.13
"$AMARKEN_ROOT/.venv/bin/python" "$AMARKEN_ROOT/cloud/run.py" --preflight

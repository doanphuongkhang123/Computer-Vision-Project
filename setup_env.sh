#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi

  export PATH="$HOME/.local/bin:$PATH"
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi

  if command -v curl >/dev/null 2>&1; then
    echo "Installing uv into user space at ~/.local/bin (no sudo)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  elif command -v wget >/dev/null 2>&1; then
    echo "Installing uv into user space at ~/.local/bin (no sudo)..."
    wget -qO- https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi

  command -v uv >/dev/null 2>&1
}

if [ ! -x ".venv/bin/python" ]; then
  rm -rf .venv
  if ensure_uv; then
    uv venv --python "$PYTHON_BIN" .venv
  else
    echo "uv is not available and could not be installed automatically." >&2
    echo "Install uv without sudo, then rerun this script:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
    exit 1
  fi
fi

if ensure_uv; then
  uv pip install --python .venv/bin/python -r requirements.txt
else
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi

echo "Environment ready: $(pwd)/.venv"

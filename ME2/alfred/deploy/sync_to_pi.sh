#!/usr/bin/env bash
set -euo pipefail

# Update an already initialized Raspberry Pi without replacing its private
# credentials, virtual environment, service state, or persistent application data.
# Usage: ./deploy/sync_to_pi.sh USER@HOST

ALFRED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-}"

fail() {
  echo "error: $*" >&2
  exit 2
}

if [[ ! "$TARGET" =~ ^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$ ]]; then
  echo "Usage: $0 USER@HOST" >&2
  exit 2
fi

command -v rsync >/dev/null 2>&1 || fail \
  "rsync is required on the development machine."
command -v ssh >/dev/null 2>&1 || fail \
  "ssh is required on the development machine."

ssh "$TARGET" '
  failed=0

  require_path() {
    if test "$1" "$2"; then
      return
    fi
    echo "missing: $3" >&2
    failed=1
  }

  require_command() {
    if command -v "$1" >/dev/null 2>&1; then
      return
    fi
    echo "missing command: $1" >&2
    failed=1
  }

  require_path -d "$HOME/alfred" "$HOME/alfred directory"
  require_path -f "$HOME/alfred/.env" "$HOME/alfred/.env"
  require_path -x "$HOME/alfred/.venv/bin/python" "$HOME/alfred/.venv/bin/python"
  require_command rsync
  require_command systemctl

  if ! command -v uv >/dev/null 2>&1 \
    && ! test -x "$HOME/.local/bin/uv" \
    && ! test -x "$HOME/.cargo/bin/uv"; then
    echo "missing command: uv (also checked ~/.local/bin/uv and ~/.cargo/bin/uv)" >&2
    failed=1
  fi

  exit "$failed"
' || fail \
  "the Raspberry Pi prerequisites listed above are incomplete. Use copy_to_pi.sh only if the Alfred directory or .env is missing."

echo "Stopping Alfred before updating its Python files"
ssh "$TARGET" 'systemctl --user stop alfred.service 2>/dev/null || true'

echo "Synchronizing updated Alfred runtime files to $TARGET:alfred/"
rsync -az --progress \
  --exclude='.DS_Store' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  "$ALFRED_ROOT/.env.example" \
  "$ALFRED_ROOT/README.md" \
  "$ALFRED_ROOT/THIRD_PARTY_NOTICES.md" \
  "$ALFRED_ROOT/pyproject.toml" \
  "$ALFRED_ROOT/uv.lock" \
  "$ALFRED_ROOT/src" \
  "$ALFRED_ROOT/config" \
  "$ALFRED_ROOT/assets" \
  "$ALFRED_ROOT/deploy" \
  "$ALFRED_ROOT/licenses" \
  "$ALFRED_ROOT/tools" \
  "$TARGET:alfred/"

echo "Refreshing locked dependencies and Raspberry Pi services"
ssh -t "$TARGET" '
  set -eu
  cd "$HOME/alfred"
  chmod 600 .env
  chmod +x deploy/initialize_pi.sh deploy/copy_to_pi.sh deploy/sync_to_pi.sh deploy/alfred
  UV_BIN="$(command -v uv 2>/dev/null || true)"
  if [ -z "$UV_BIN" ] && [ -x "$HOME/.local/bin/uv" ]; then
    UV_BIN="$HOME/.local/bin/uv"
  fi
  if [ -z "$UV_BIN" ] && [ -x "$HOME/.cargo/bin/uv" ]; then
    UV_BIN="$HOME/.cargo/bin/uv"
  fi
  [ -n "$UV_BIN" ] || {
    echo "error: uv is unavailable on the Raspberry Pi" >&2
    exit 2
  }
  "$UV_BIN" sync --frozen --no-dev
  ./deploy/initialize_pi.sh --preserve-reminders
'

echo
echo "Raspberry Pi update completed. Persistent credentials and data were preserved."
echo "Check services with: ssh $TARGET alfred status"
echo "Follow logs with:    ssh -t $TARGET alfred logs"

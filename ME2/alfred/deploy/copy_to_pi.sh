#!/usr/bin/env bash
set -euo pipefail

# Copy only Alfred runtime files from macOS/Linux to a Raspberry Pi.
# Usage: ./deploy/copy_to_pi.sh USER@HOST

ALFRED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-}"

if [[ ! "$TARGET" =~ ^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$ ]]; then
  echo "Usage: $0 USER@HOST" >&2
  exit 2
fi

[[ -f "$ALFRED_ROOT/.env" ]] || {
  echo "error: $ALFRED_ROOT/.env is required before deployment." >&2
  exit 2
}

command -v rsync >/dev/null 2>&1 || {
  echo "error: rsync is required on the development machine." >&2
  exit 2
}
if ! ssh "$TARGET" 'command -v rsync >/dev/null 2>&1'; then
  echo "error: rsync is not installed on the Raspberry Pi." >&2
  echo "Complete the manual packages in deploy/RASPBERRY_PI_SETUP.md first." >&2
  exit 2
fi
ssh "$TARGET" 'install -d -m 700 "$HOME/alfred" "$HOME/alfred/secrets"'

echo "Copying Alfred runtime files to $TARGET:alfred/"
rsync -az --progress \
  --exclude='.DS_Store' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  "$ALFRED_ROOT/.env" \
  "$ALFRED_ROOT/.env.example" \
  "$ALFRED_ROOT/README.md" \
  "$ALFRED_ROOT/THIRD_PARTY_NOTICES.md" \
  "$ALFRED_ROOT/pyproject.toml" \
  "$ALFRED_ROOT/uv.lock" \
  "$ALFRED_ROOT/src" \
  "$ALFRED_ROOT/secrets" \
  "$ALFRED_ROOT/config" \
  "$ALFRED_ROOT/assets" \
  "$ALFRED_ROOT/deploy" \
  "$ALFRED_ROOT/licenses" \
  "$ALFRED_ROOT/tools" \
  "$TARGET:alfred/"

ssh "$TARGET" \
  'chmod 600 "$HOME/alfred/.env" && chmod +x "$HOME/alfred/deploy/initialize_pi.sh" "$HOME/alfred/deploy/alfred"'

echo
echo "Upload completed. Complete the manual setup on the Raspberry Pi:"
echo "  ~/alfred/deploy/RASPBERRY_PI_SETUP.md"
echo "Then initialize and start Alfred:"
echo "  cd ~/alfred && ./deploy/initialize_pi.sh"

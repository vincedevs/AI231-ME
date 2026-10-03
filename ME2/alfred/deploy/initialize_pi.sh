#!/usr/bin/env bash
set -euo pipefail

# Initialize Alfred after its operating-system packages, Python environment,
# private Google Chat webhook and Wayne Manor URL are configured.
# This script configures local state and starts services; it installs no software.

ALFRED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALFRED_TIMEZONE="${ALFRED_TIMEZONE:-Asia/Manila}"
CURRENT_USER="$(id -un)"
PYTHON_BIN="$ALFRED_ROOT/.venv/bin/python"
RESET_REMINDERS=true

if [[ "${1:-}" == "--preserve-reminders" ]]; then
  RESET_REMINDERS=false
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--preserve-reminders]" >&2
  exit 2
fi

fail() {
  echo "error: $*" >&2
  exit 2
}

[[ "${EUID}" -ne 0 ]] || fail "Run this script as the normal Alfred user, not with sudo."
case "$(uname -m)" in
  aarch64|arm64) ;;
  *) fail "Alfred requires 64-bit Raspberry Pi OS. Detected architecture: $(uname -m)" ;;
esac
[[ "$(uname -s)" == "Linux" ]] || fail "This script must run on Raspberry Pi OS/Linux."
[[ "$ALFRED_ROOT" == "$HOME/alfred" ]] || \
  fail "Copy Alfred to $HOME/alfred. The supplied user services use that location."

echo "[1/6] Validating the manual installation"
for command in sudo systemctl timedatectl systemd-run lsusb arecord aplay wpctl; do
  command -v "$command" >/dev/null 2>&1 || fail "Required command is unavailable: $command"
done
[[ -x "$PYTHON_BIN" ]] || \
  fail "Python environment missing; complete the Python section in deploy/RASPBERRY_PI_SETUP.md."
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || \
  fail "Alfred's virtual environment must use Python 3.12."

[[ -f "$ALFRED_ROOT/.env" ]] || fail "Copy .env to $ALFRED_ROOT/.env first."
chmod 600 "$ALFRED_ROOT/.env"
"$PYTHON_BIN" "$ALFRED_ROOT/tools/test_google_chat.py" check || \
  fail "Google Chat configuration is incomplete. Review ALFRED_GOOGLE_CHAT_WEBHOOK_URL in .env."
systemctl --user stop alfred.service 2>/dev/null || true
# Remove obsolete Alfred-owned service definitions. External account data and
# an optional manually installed Soloist executable are left untouched.
systemctl --user disable --now spotify-soloist.service 2>/dev/null || true
rm -f "$HOME/.config/systemd/user/spotify-soloist.service"
rm -f "$ALFRED_ROOT/deploy/spotify-soloist.service"
systemctl --user unset-environment ALFRED_MUSIC_BACKEND 2>/dev/null || true
systemctl --user disable --now baresip.service 2>/dev/null || true
rm -f "$HOME/.config/systemd/user/baresip.service"
rm -f "$ALFRED_ROOT/deploy/baresip.service"

echo "[2/6] Starting the user audio services"
systemctl --user daemon-reload
systemctl --user enable --now pipewire.socket pipewire-pulse.socket wireplumber.service
wpctl status >/dev/null || fail "PipeWire/WirePlumber did not become ready."

echo "[3/6] Initializing Alfred's local state"
sudo timedatectl set-timezone "$ALFRED_TIMEZONE"
install -d -m 700 "$ALFRED_ROOT/data"
install -d -m 700 "$ALFRED_ROOT/secrets"
if [[ "$RESET_REMINDERS" == true ]]; then
  "$PYTHON_BIN" -m alfred --reset-db
else
  "$PYTHON_BIN" -m alfred --init-db
fi
"$PYTHON_BIN" -m alfred --self-check

echo "[4/6] Detecting and configuring ReSpeaker Lite"
lsusb | grep -qiE '2886:0019|ReSpeaker' || fail "ReSpeaker Lite was not detected on USB."
echo "ReSpeaker Lite detected on USB."
arecord -l || true
aplay -l || true
"$PYTHON_BIN" -m alfred --list-devices
"$PYTHON_BIN" -m alfred --configure-audio-match ReSpeaker

sink_line="$(wpctl status | sed -n '/Sinks:/,/Sources:/p' | grep -im1 'ReSpeaker' || true)"
source_line="$(wpctl status | sed -n '/Sources:/,/Filters:/p' | grep -im1 'ReSpeaker' || true)"
sink_id="$(printf '%s' "$sink_line" | grep -oE '[0-9]+\.' | head -n 1 | tr -d '.' || true)"
source_id="$(printf '%s' "$source_line" | grep -oE '[0-9]+\.' | head -n 1 | tr -d '.' || true)"
[[ -n "$sink_id" && -n "$source_id" ]] || \
  fail "PipeWire did not expose both ReSpeaker source and sink nodes."
wpctl set-default "$sink_id"
wpctl set-default "$source_id"
wpctl set-volume "$sink_id" 50%
echo "ReSpeaker selected as the PipeWire default source and sink at 50% initial volume."

echo "[5/6] Installing Alfred's user service definitions"
install -d -m 700 "$HOME/.config/systemd/user"
install -m 644 "$ALFRED_ROOT/deploy/alfred.service" \
  "$HOME/.config/systemd/user/alfred.service"
sudo install -m 755 "$ALFRED_ROOT/deploy/alfred" /usr/local/bin/alfred
sudo loginctl enable-linger "$CURRENT_USER"
systemctl --user daemon-reload

echo "[6/6] Enabling and starting Alfred's services"
systemctl --user enable --now alfred.service
systemctl --user is-active --quiet alfred.service || fail \
  "Alfred failed to start. Inspect: journalctl --user -u alfred.service -n 100 --no-pager"

echo
echo "Alfred initialization completed."
echo "Service state: systemctl --user status alfred.service --no-pager"
echo "Live log:      journalctl --user -u alfred.service -f"
echo "Message check:  $PYTHON_BIN $ALFRED_ROOT/tools/test_google_chat.py check"
echo "Stop all:      alfred stop"
echo "Start all:     alfred run"

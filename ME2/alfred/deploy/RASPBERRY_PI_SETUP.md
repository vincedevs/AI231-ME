# Raspberry Pi 5 setup

This guide separates manual software installation from Alfred initialization.
The shell scripts in this directory do not install operating-system or Python
packages. `copy_to_pi.sh` copies the runtime; `initialize_pi.sh` initializes
local state, installs user-service definitions, and starts Alfred's services.

The Wayne Manor API is a separately operated external service. Nothing in this
guide installs, starts, stops, or supervises it.

## 1. Prepare the Raspberry Pi

Use a Raspberry Pi 5 with the current 64-bit Raspberry Pi OS based on Debian
Trixie and a normal user account with `sudo` access. Enable SSH if deployment
will be performed from another machine.

Confirm the operating system and architecture before continuing:

```bash
grep -E '^(PRETTY_NAME|VERSION_CODENAME)=' /etc/os-release
uname -m
getconf GNU_LIBC_VERSION
```

`uname -m` must report `aarch64`. Prefer a fresh current image instead of an
in-place major-version upgrade from Bookworm to Trixie.

References:

- [Install Raspberry Pi OS](https://www.raspberrypi.com/documentation/computers/getting-started.html)
- [Raspberry Pi OS versions and editions](https://www.raspberrypi.com/documentation/computers/os.html)
- [Enable and use SSH](https://www.raspberrypi.com/documentation/computers/remote-access.html)
- [ReSpeaker Lite hardware and firmware](https://github.com/respeaker/ReSpeaker_Lite/)

Connect the ReSpeaker Lite over USB. Its USB firmware must expose simultaneous
capture and playback before Alfred can select it as a sound device.

## 2. Install native packages manually

Run these commands on the Raspberry Pi:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  alsa-utils \
  ca-certificates \
  curl \
  libgomp1 \
  libportaudio2 \
  libsndfile1 \
  pipewire-audio \
  rsync \
  usbutils
```

These provide ALSA diagnostics, TLS certificates, the native
audio libraries used by Python, the PipeWire/WirePlumber audio stack used for
ReSpeaker routing and volume control, file transfer, and USB-device inspection. `pipewire-audio` is
included explicitly because Raspberry Pi OS Lite does not include a desktop
audio server by default.

Package references:

- [Debian `alsa-utils`](https://packages.debian.org/stable/alsa-utils)
- [Debian `ca-certificates`](https://packages.debian.org/stable/ca-certificates)
- [Debian `curl`](https://packages.debian.org/stable/curl)
- [Debian `libgomp1`](https://packages.debian.org/stable/libgomp1)
- [Debian `libportaudio2`](https://packages.debian.org/stable/libportaudio2)
- [Debian `libsndfile1`](https://packages.debian.org/stable/libsndfile1)
- [Debian `pipewire-audio`](https://packages.debian.org/stable/pipewire-audio)
- [Raspberry Pi audio configuration](https://www.raspberrypi.com/documentation/computers/configuration.html#change-audio-output)
- [Debian `rsync`](https://packages.debian.org/stable/rsync)
- [Debian `usbutils`](https://packages.debian.org/stable/usbutils)

After installing the packages, verify that Linux sees both sides of the
ReSpeaker USB audio device:

```bash
lsusb
arecord -l
aplay -l
```

Do not continue unless ReSpeaker appears as both a capture and playback device.
If it does not, follow the official ReSpeaker Lite firmware instructions linked
in section 1, reconnect the device, and repeat these checks.

## 3. Install `uv` manually

Alfred's lock file is built with `uv`. The following pins the installer version
used by this repository and keeps the executable under the current user's home
directory. Review downloaded scripts before running them. It also explicitly
adds `~/.local/bin` to the login PATH, so later commands and future SSH sessions
can invoke `uv` directly.

```bash
curl -fLsS https://astral.sh/uv/0.11.25/install.sh -o /tmp/uv-install.sh
less /tmp/uv-install.sh
sh /tmp/uv-install.sh

UV_PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'
grep -qxF "$UV_PATH_LINE" "$HOME/.profile" || printf '%s\n' "$UV_PATH_LINE" >> "$HOME/.profile"
source "$HOME/.profile"

command -v uv
uv --version
```

`command -v uv` must print a path, normally `/home/USER/.local/bin/uv`. If it
does not, stop here and inspect `~/.profile` before continuing.

References:

- [`uv` installation](https://docs.astral.sh/uv/getting-started/installation/)
- [`uv` Python-version management](https://docs.astral.sh/uv/concepts/python-versions/)

## 4. Prepare CALL and MESSAGE backends manually

### 4.1 Optional real CALL backend: Linphone

The default demonstration backend is the Wayne Manor telephone and requires no
telephony package on the Pi. To enable Alfred's real SIP option, install the
Debian Linphone command-line client and confirm that the Alfred SIP account is
already registered:

```bash
sudo apt-get install -y linphone-cli
linphonecsh init
linphonecsh status register
```

Use the Linphone account configuration that you have already verified with a
manual `linphonecsh dial`. Put only the destination SIP URI in Alfred's private
environment:

```dotenv
ALFRED_LINPHONE_DESTINATION="sip:recipient@sip.linphone.org"
```

Run `alfred run --call-backend linphone` for that backend. Plain `alfred run`
clears the override and returns to the configured default (`wayne_manor`). The
CALL intent is slotless, so the destination is fixed rather than inferred from
speech.

References:

- [Linphone project](https://www.linphone.org/)
- [Debian `linphone-cli` package](https://packages.debian.org/stable/linphone-cli)

### 4.2 Google Chat messaging

Create an incoming webhook in the Google Chat space where Alfred should post.
Copy its URL into `ALFRED_GOOGLE_CHAT_WEBHOOK_URL` in `alfred/.env`. Treat the
URL as a credential and keep it private. Google Chat incoming webhooks always
post into their configured space. VCM-only mode posts generic message text
without a recipient label. ASR mode adds the captured recipient and content as
visible `To:` and `Message:` lines; it does not route to another Chat space.

References:

- [Create an incoming webhook](https://developers.google.com/workspace/chat/quickstart/webhooks)
- [Incoming webhook reference](https://developers.google.com/workspace/chat/quickstart/webhooks#update_the_webhook)

## 5. Prepare Alfred's private environment

On the development machine, ensure `alfred/.env` contains the real private
values before copying:

```dotenv
ALFRED_WAYNE_MANOR_URL="http://wayne-manor-host:8765/api/v1"
ALFRED_LINPHONE_DESTINATION="sip:recipient@sip.linphone.org"
ALFRED_GOOGLE_CHAT_WEBHOOK_URL="https://chat.googleapis.com/v1/spaces/.../messages?key=...&token=..."
```

In VCM-only mode, Alfred posts one of the five non-sensitive fallback texts in
`config/default.json` without a recipient label. With
`alfred run --enable-asr`, the spoken recipient and dictated message replace
that fallback behavior.

Set `ALFRED_WAYNE_MANOR_URL` to an address the Pi can reach. If the API runs on
another computer, use that computer's LAN IP address or resolvable hostname;
`127.0.0.1` and `localhost` would incorrectly point back to the Pi. If the API
really runs as a separate process on the Pi itself, remove this line and Alfred
uses its loopback default. Alfred never starts or supervises Wayne Manor.

After installation, update a changing Wayne Manor address and restart Alfred
with one command:

```bash
alfred run --set-wayne-manor-url http://192.168.1.9:8765/api/v1
```

Use the laptop's current address as seen from the Pi. When available, a stable
mDNS hostname or router DHCP reservation avoids changing this value between
networks.

Reminder database maintenance is explicit:

```bash
# Create the database if absent, or validate its existing schema.
alfred --init-db

# Delete all reminders and recreate an empty database.
alfred stop
alfred --reset-db
alfred run
```

The copy script transfers `.env` through SSH and sets owner-only permissions
on the Pi. Never commit the file or paste the webhook URL into logs.

## 6. Copy the runtime to the Pi

Moonshine v2 Tiny English is optional. Alfred's default service uses only the
VCM for command understanding and does not require these files. To test the
optional `--enable-asr` mode, install the model on the development machine
before copying. Model binaries remain outside Git, while the installer records
their source and hashes:

```bash
cd /path/to/ME2/alfred
python3.12 tools/install_clarification_asr.py
```

Official model source: [Sherpa-ONNX Moonshine v2 Tiny English archive](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2).

Validate the default VCM-only runtime:

```bash
cd /path/to/ME2/alfred
.venv/bin/python -m alfred --self-check
```

When optional ASR was installed, validate it explicitly as well:

```bash
.venv/bin/python -m alfred --enable-asr --self-check
```

Do not upload if the self-check reports a missing model or checksum mismatch.
After it passes, run the copy script from the same directory:

```bash
./deploy/copy_to_pi.sh USER@RASPBERRY_PI_HOST
```

The script copies the runtime code, configuration, locked dependency metadata,
models, audio assets, licenses, deployment files, and `.env`. It deliberately
excludes datasets, experiment outputs, tests, developer tools, local caches,
the development virtual environment, and the development reminder database.

## 7. Install Python 3.12 and locked dependencies manually

Run this on the Raspberry Pi after the copy completes. `uv python install`
installs the managed Python 3.12 interpreter, `uv python pin` selects it for
this project, and `uv sync` automatically creates the isolated
`~/alfred/.venv` environment before installing the locked dependencies. Do not
run `uv venv` separately.

```bash
cd "$HOME/alfred"
uv python install 3.12
uv python pin 3.12
uv sync --frozen --no-dev
.venv/bin/python --version
```

`--frozen` requires the committed `uv.lock` and prevents dependency resolution
from silently changing the environment. The final command must report Python
3.12.x from `~/alfred/.venv`; `initialize_pi.sh` rejects a missing environment
or any other Python minor version.

## 8. Initialize and start Alfred

Run the initializer as the normal user, not with `sudo`:

```bash
cd "$HOME/alfred"
./deploy/initialize_pi.sh
```

It performs only local initialization and service startup:

1. validates the manually installed programs, Python 3.12 environment, core
   models and private configuration without contacting external services;
2. enables and starts PipeWire, its PulseAudio compatibility socket, and
   WirePlumber;
3. sets the default `Asia/Manila` timezone and resets Alfred's SQLite reminder
   database to an empty initialized state;
4. requires and selects ReSpeaker Lite as input and output, then limits its
   system output to 50%;
5. removes obsolete Alfred-owned service definitions, installs `alfred.service`,
   and installs the `/usr/local/bin/alfred` control command;
6. enables and starts Alfred.

To use another timezone:

```bash
ALFRED_TIMEZONE=Region/City ./deploy/initialize_pi.sh
```

The services use systemd's user manager and linger so they can start without an
interactive login. References:

- [Debian `systemctl` manual](https://manpages.debian.org/stable/systemd/systemctl.1.en.html)
- [Debian `loginctl` manual](https://manpages.debian.org/stable/systemd/loginctl.1.en.html)

After this one-time initialization, control the entire application with one
command:

```bash
alfred run       # start Alfred
alfred run --enable-asr  # start Alfred with optional Moonshine follow-ups
alfred run --call-backend linphone  # use the configured SIP destination
alfred run --call-backend wayne_manor  # ring the simulated telephone
alfred stop      # stop Alfred
alfred restart   # restart Alfred
alfred restart --enable-asr  # restart with optional Moonshine follow-ups
alfred status    # show the service state
alfred logs      # follow Alfred's journal; Ctrl-C only exits the log viewer
```

These commands use the real configured CALL backend, Google Chat messaging,
systemd timer/alarm, Open-Meteo, Wayne Manor, SQLite, and system-volume backends
because the installed service does not pass the development-only `--mock`
option. Wayne Manor remains a separately operated service and is not started by
Alfred.

The command deliberately leaves PipeWire and WirePlumber running because they
are shared operating-system audio services, not Alfred-owned processes.

## 9. Verify and observe the services

```bash
systemctl --user status alfred.service --no-pager
journalctl --user -u alfred.service -f
```

Additional checks:

```bash
cd "$HOME/alfred"
.venv/bin/python -m alfred --self-check
.venv/bin/python -m alfred --list-devices
arecord -l
aplay -l
wpctl status
```

The installed service runs in VCM-only mode unless ASR is explicitly enabled:

```bash
alfred run --enable-asr
```

Return to the VCM-only service with `alfred run`. The controller restarts the
Alfred process when changing modes, so the requested setting takes effect even
when the service was already active.

Confirm that the selected ReSpeaker sink starts at the expected level:

```bash
wpctl get-volume @DEFAULT_AUDIO_SINK@
```

If this reports a different level than `0.50`, set its initial level. This does
not impose a persistent ceiling; Alfred's volume commands retain the normal
0–100% range:

```bash
wpctl set-volume @DEFAULT_AUDIO_SINK@ 50%
```

Make a short microphone recording and play it through the selected default
output. Speak only non-sensitive test audio because this writes a temporary WAV
file:

```bash
arecord --device=default --duration=3 --format=S16_LE \
  --rate=16000 --channels=1 /tmp/alfred-microphone-test.wav
aplay /tmp/alfred-microphone-test.wav
rm -f /tmp/alfred-microphone-test.wav
```

Validate the Google Chat webhook format locally without sending a message:

```bash
cd "$HOME/alfred"
.venv/bin/python tools/test_google_chat.py check
```

To test delivery, send one explicit message to the configured Chat space:

```bash
.venv/bin/python tools/test_google_chat.py message \
  --text "Alfred test message." --confirm-live

.venv/bin/python tools/test_google_chat.py message \
  --recipient Dad --text "I will be home at six." --confirm-live
```

The test confirms that Google Chat accepted the webhook request.

When Alfred starts, it probes Wayne Manor, Open-Meteo, and Google Chat once and
logs each service as `available` or `unavailable`. An unavailable service is
then rejected immediately instead of being contacted again for every command.
After correcting an endpoint, credential, network, or firewall problem, refresh
the in-memory availability flags with:

```bash
alfred restart
```

Verify Wayne Manor from the Pi independently of Alfred. Replace the example
host with the value placed in `.env`; a connection refusal or timeout must be
fixed in the external API, host firewall, bind address, or LAN routing:

```bash
curl --verbose --max-time 3 http://wayne-manor-host:8765/api/v1/
systemctl --user restart alfred.service
```

Finally, confirm that linger restores the services without an interactive
login. Reboot the Pi, reconnect over SSH, and repeat the service-status command:

```bash
sudo reboot
```

After the Pi returns:

```bash
systemctl --user status alfred.service --no-pager
```

## Updating the deployment

For an already initialized Pi, run the dedicated update command from the
development machine:

```bash
./deploy/sync_to_pi.sh USER@RASPBERRY_PI_HOST
```

It preserves the Pi's private `.env`, virtual environment location, and
reminder database. It copies changed runtime files, runs
`uv sync --frozen --no-dev`, reapplies configuration and service definitions,
and restarts Alfred. The ordinary
`initialize_pi.sh` still resets reminders; the sync script intentionally calls
its update-only `--preserve-reminders` mode.

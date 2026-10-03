# Alfred

Alfred is the isolated Raspberry Pi 5 runtime for the project's three-stage
voice-assistant pipeline:

```text
wake word → Silero command capture → VCM → policy → action → idle
```

Wake-word, VAD, VCM, and TTS inference run locally. Optional clarification ASR
also runs locally, but is disabled unless the user explicitly passes
`--enable-asr`. Version 1.0 uses a small, explicit mixture of local,
operating-system, and reviewed API actions.

## Current capabilities

- "Hey Alfred" and "I'm Batman" wake-word models;
- an ONNX-only openWakeWord-compatible frontend that supports Python 3.12;
- configurable wake thresholds, repeated-hit gate, and cooldown;
- direct Silero VAD ONNX inference with pre-roll and trailing-silence capture;
- an explicitly selectable mock VCM for deterministic application testing;
- the selected raw-waveform Tiny Conformer ONNX VCM, including its log-Mel frontend;
- calibrated unsupported and low-confidence decisions;
- a pre-inference finite-audio and minimum-RMS safety gate;
- strict validation of the 19-intent, six-slot contract;
- optional, explicitly enabled Moonshine v2 clarification for missing, invalid,
  or uncertain slots;
- explicit announcement-before-action ordering;
- Open-Meteo weather for a location requested in a spoken follow-up, with
  Quezon City as the fallback;
- local time plus Spotify playback-volume actions through Wayne Manor;
- output-rate adaptation for USB sound cards such as ReSpeaker Lite;
- persistent local reminders backed by SQLite;
- two-turn dictated messages posted to a configured Google Chat space;
- WayneManor API controls for lights, brightness, color, and thermostat setpoint;
- one-shot Raspberry Pi systemd timers and alarms;
- selectable CALL execution through a configured Linphone SIP destination or
  the local Wayne Manor ringing-telephone demo;
- checksummed runtime model bundle;
- synthesized xylophone feedback cues with reproducible generation code;
- offline Piper VITS speech synthesis using the British English
  `en_GB-alan-medium` voice for action announcements and responses;
- Raspberry Pi `systemd` deployment files.

## Environment

Alfred requires Python 3.12. The committed lock file is the reproducible setup:

```bash
uv sync --frozen --extra dev
source .venv/bin/activate
```

Raspberry Pi software is installed manually. Follow the complete, source-linked
instructions in [`deploy/RASPBERRY_PI_SETUP.md`](deploy/RASPBERRY_PI_SETUP.md).
After installation, `deploy/initialize_pi.sh` initializes local state and starts
Alfred; it does not install software.

Persist a new Wayne Manor address and restart Alfred in one command:

```bash
alfred run --set-wayne-manor-url http://192.168.1.9:8765/api/v1
```

The command updates only `ALFRED_WAYNE_MANOR_URL` in the private `.env` file;
other credentials remain unchanged. A resolvable hostname such as
`http://wayne-manor.local:8765/api/v1` is preferable to a changing DHCP address
when the network supports mDNS.

Initialize or validate the reminder database without deleting reminders:

```bash
alfred --init-db
```

Clear every reminder and recreate the schema:

```bash
alfred stop
alfred --reset-db
alfred run
```

## Verify the package

```bash
python -m alfred --self-check
python -m pytest
```

The self-check verifies Python dependencies, model checksums, configuration,
every feedback asset, actual silence inference through the wake-word and VAD
graphs, real TTS synthesis, reminder-database initialization, and availability
of a supported volume backend. The volume check is detection-only: it does not
change the current volume. The self-check also executes the real VCM when
`vcm.mode` is set to `onnx`. To verify the optional Moonshine installation too,
run `python -m alfred --enable-asr --self-check` or
`python -m alfred --weather-asr --self-check`.

## VCM-only and optional-ASR modes

Alfred defaults to the academically compliant VCM-only path, including when
managed as a Raspberry Pi service:

```bash
alfred run
```

In this mode, openWakeWord activates Alfred, Silero VAD captures the command,
and the VCM is the only component that interprets command semantics. Missing or
invalid required VCM slots are rejected. Weather uses the configured default
location. Because message content is not a canonical VCM slot, `MESSAGE` sends
one of five configured generic fallback texts. This fallback applies only when
ASR is disabled.

Enable the optional Moonshine follow-up experience in the same background
service interface:

```bash
alfred run --enable-asr
```

The controller passes this mode into `alfred.service` through systemd's user
manager environment and restarts Alfred so the requested mode always takes
effect. Plain `alfred run` clears that setting and returns to VCM-only mode.
The mode does not affect Wayne Manor media, the selected call backend, or
Google Chat message configuration.

Weather-location transcription has its own switch. By default, it follows the
general ASR setting; enable it alone with `alfred run --weather-asr`, or turn
it off while keeping other ASR follow-ups enabled with
`alfred run --enable-asr --no-weather-asr`. When weather ASR is off, Alfred
does not ask for a location and uses the configured Quezon City default.

## VCM benchmark event log

When Alfred runs, it writes one append-only JSON Lines file per process under
`~/vcm_benchmark/`. This is compatible with the external
[`vcm-benchmark`](https://github.com/airimonda/vcm-benchmark) harness. A
primary `Hey Alfred` event is logged as a wake event, followed by exactly one
command result after VCM inference. Rejected predictions are recorded as
`OUT_OF_SCOPE`; Alfred never writes its rejected argmax as an executable
intent. The Batman easter-egg wake word produces neither benchmark event nor
command result.

`infer_ms` measures only the synchronous `vcm.predict()` call, including the
embedded ONNX feature frontend and CTC slot decoding. `audio_ms` is the
captured 16 kHz command waveform duration. Neither value includes wake-word
detection, VAD capture, clarification ASR, TTS, or action/backend time.

### Physical benchmark mode

Use Alfred's explicit benchmark mode for the physical
laptop-speaker → Raspberry-Pi/ReSpeaker-microphone evaluation:

```bash
alfred run --benchmark
```

For a foreground diagnostic run, the equivalent command is:

```bash
python -m alfred --benchmark
```

This mode retains real microphone input, OpenWakeWord detection, Silero VAD
command capture, VCM inference, rejection logic, and the JSON Lines event log.
Immediately after the VCM result is logged, Alfred returns to wake-word
listening. It deliberately disables Moonshine follow-ups, feedback sounds,
TTS, Spotify interaction control, policy/action execution, reminder writes,
and all local or network backends. The Batman wake word is ignored without
creating a benchmark record.

Run the external harness from the laptop with its log directory set to the
same production path:

```bash
python benchmark.py --host raspy@raspy.local --log-dir '~/vcm_benchmark'
```

This is an end-to-end physical audio benchmark, not direct WAV injection or a
model-only speed test. Its response timing covers the real speaker/microphone
path through the appearance of Alfred's command log, while Alfred's `infer_ms`
field remains limited to VCM preprocessing plus ONNX prediction and
`audio_ms` remains the captured waveform duration.

For an isolated local test destination, set `ALFRED_BENCHMARK_LOG_DIR` before
starting Alfred. The production default must remain `~/vcm_benchmark/` for the
benchmark harness.

## Compare MacBook and Raspberry Pi performance

Run the same local-inference benchmark on both devices. Stop Alfred first on
the Pi so its continuously running models do not compete with the benchmark:

```bash
python -m alfred stop
python tools/benchmark_device.py --label raspberry_pi_5 --warmup 5 --runs 30
```

On the MacBook, run the same workload from the Alfred environment:

```bash
python tools/benchmark_device.py --label macbook --warmup 5 --runs 30
```

The generated workload is identical across architectures and is suitable for
latency comparison, but not accuracy analysis. For a functional comparison,
record or select one real command as a **16 kHz mono WAV**, copy that exact file
to both devices, and add the same argument to both commands:

```bash
python tools/benchmark_device.py \
  --label raspberry_pi_5 \
  --audio benchmark_command.wav \
  --warmup 5 \
  --runs 30
```

Each run creates `benchmark.json` and `measurements.csv` under
`benchmark_results/`. The report includes cold initialization, p50/p95/p99
steady-state latency, real-time factor, whole-process peak memory, model and
workload hashes, software versions, and Raspberry Pi temperature/throttling
state when available. It tests wake-word inference, Silero VAD, the VCM, and
Piper TTS without playing audio or calling network APIs. Moonshine is measured
only when clarification ASR is explicitly enabled in the benchmark's supplied
configuration.

Copy the Pi result folder back to the MacBook, then compare the two JSON files:

```bash
python tools/compare_benchmarks.py \
  benchmark_results/macbook_TIMESTAMP/benchmark.json \
  benchmark_results/raspberry_pi_5_TIMESTAMP/benchmark.json
```

This writes a Markdown summary and comparison CSV. A candidate/baseline ratio
above `1.0` means the Raspberry Pi is slower for that latency metric. The
comparison warns when the workload hash, model manifest, warm-up count, or run
count differs. Restart Alfred after benchmarking:

```bash
python -m alfred start
```

These timings isolate compute performance. Separately demonstrate the full
ReSpeaker path with normal commands and record wake-word success, command
capture success, correct intent/action, and wall-clock response time. Do not
combine network-backed action latency with the local model benchmark.

## Run

Initialize or validate the local reminder database manually:

```bash
python -m alfred --init-db
```

This command is safe to repeat and never deletes existing reminders. Normal
application startup also performs the same initialization automatically before
loading the audio and ML models. To deliberately remove every reminder while
preserving the schema, run `python -m alfred --reset-db`. The Raspberry Pi
initializer uses this destructive reset so every full initialization starts
with an empty reminder database.

List audio devices:

```bash
python -m alfred --list-devices
```

Select a connected ReSpeaker for both capture and playback without manually
copying PortAudio indexes:

```bash
python -m alfred --configure-audio-match ReSpeaker
```

The Pi initializer runs this selection before starting the service and stops if
the ReSpeaker does not provide both input and output channels. When PipeWire is
available, the initializer also selects the ReSpeaker as the operating system's
default source and sink so Spotify and volume actions use the same
hardware.

Run continuously with the packaged real VCM (the default):

```bash
python -m alfred
```

Select the real or mock VCM for one run without editing configuration:

```bash
python -m alfred --vcm-mode onnx
python -m alfred --vcm-mode mock
```

The default packaged model is the current candidate. The previous `3a2a`
candidate remains available for a controlled runtime comparison without
replacing the default model or editing the manifest:

```bash
alfred run --vcm-variant 3a2a
```

Plain `alfred run` clears that temporary service setting and returns to the
current candidate. The selected variant's own metadata, calibration, and
confidence thresholds remain intact.

Exit after one successful "Hey Alfred" command:

```bash
python -m alfred --once
```

Choose a mock outcome without editing configuration:

```bash
python -m alfred --once --mock-intent LIGHT_ON
python -m alfred --once --mock-decision unsupported
```

Any `--mock-intent`, `--mock-decision`, or `--mock-slot` argument selects mock
mode automatically. Mock predictions still pass through the same action
allowlist and policy gates as real predictions.

Use `--headless-feedback` to print announcements without playing sound. This is
useful when exercising the state flow without speakers, although a microphone
is still required for live wake-word and VAD operation.

Use `--no-tts` to retain the WAV feedback cues while printing action text
instead of synthesizing it.

## Real VCM handoff

New experiment runs write `deployment_metadata.json` beside `model.onnx`.
Install a completed experiment transactionally with:

```bash
python tools/install_vcm.py --experiment /path/to/experiment --dry-run
python tools/install_vcm.py --experiment /path/to/experiment --activate
```

The dry run validates schema, calibration metadata, ONNX inputs and outputs,
and verifies that the application safety gate rejects digital silence without
changing Alfred. Installation stores the
model in a content-addressed directory and atomically switches the checksummed
manifest. `--activate` also changes `vcm.mode` to `onnx` and synchronizes the
application intent-confidence floor.

Experiment directories created before automatic metadata export are supported:
the installer reconstructs their metadata from `config.json` and `metrics.json`.
The recorded values were fitted on validation and held fixed during test
evaluation. `config/vcm_metadata.example.json` documents the resulting fields:

- the 19 labels in output order;
- the intent-to-slot mapping;
- the CTC token order;
- rejection strategy;
- validation-fitted temperature and scope threshold;
- minimum intent confidence selected using validation data only; and
- minimum waveform length.

`tools/package_models.py` is retained for constructing a complete bundle from
unpackaged wake-word and VAD artifacts. Normal VCM swaps should use
`tools/install_vcm.py`, so those stable artifacts do not need to be supplied
again.

To override the preconfigured intent-confidence floor during a reviewed
deployment:

```bash
python tools/install_vcm.py \
  --experiment /path/to/experiment \
  --minimum-intent-confidence 0.75 \
  --activate
```

Only use an override selected from that model's validation predictions. Do not
reuse a threshold calibrated for another checkpoint. The currently packaged
Tiny Conformer uses the ONNX Runtime validation-only operating point recorded
in its metadata: minimum intent confidence `0.866007924079895` and scope
threshold `0.9999032967447421`. No official test or Raspberry Pi holdout data
was used to select these thresholds.

## Configuration

Runtime settings are in `config/default.json`. Important values include:

- audio devices and block size;
- separate feedback and TTS gains (15% each by default), relative to system volume;
- wake thresholds and required hits;
- VAD threshold and endpointing durations;
- four-second speech-start timeout after the listening cue;
- twelve-second maximum command duration;
- mock versus ONNX VCM mode;
- packaged VCM intent-confidence floor (`0.866007924079895`);
- minimum command-waveform RMS (`0.0001`, or -80 dBFS, by default);
- the explicit action-intent allowlist;
- weather follow-up prompt, fallback location and coordinates, timeout, and
  cache duration;
- system-volume step size and Raspberry Pi maximum;
- reminder database path and spoken-list limit;
- CALL backend, Linphone executable/destination environment, and command timeout;
- optional ASR prompts, slot-confidence gate, and Moonshine paths;
- Google Chat webhook environment-variable name, follow-up prompt, message
  limits, and timeout;
- systemd alert limits, repetitions, and timeout;
- Wayne Manor URL, device IDs, temperature unit, safe ranges, and timeout; and
- one-second startup API-probe timeout;
- post-feedback cooldown.

All paths must remain inside the `alfred` folder. Startup rejects paths that
escape it.

## Action backends

The reviewed real handlers are intentionally listed in
`config/action_contract.json`; startup rejects unreviewed or accidentally
remapped real handlers. `actions.enabled_intents` is a separate runtime safety
allowlist. All 19 canonical intents are enabled. `TIMER`, `ALARM`, and
`CREATE_REMINDER` still require their canonical slots. `TEMPERATURE`,
`BRIGHTNESS`, and `COLOR` use explicit safe application defaults when their
slot is absent, invalid, unsafe, or unsupported.

| Intent | Backend | Behavior |
| --- | --- | --- |
| `WEATHER` | Open-Meteo | Optionally asks for a location and transcribes it with Moonshine; otherwise uses the configured Quezon City default. Results are cached for ten minutes by default. |
| `TIME` | Python system clock | Speaks the host's local time. |
| `VOLUME_UP`, `VOLUME_DOWN` | Wayne Manor Spotify player | Changes laptop Spotify playback by five percentage points over 0–100%. The UI slider can set an exact level. |
| `CREATE_REMINDER`, `LIST_REMINDERS` | SQLite + optional offline ASR | Persists reminders in `data/state.sqlite3`. Listing speaks every active reminder as a separate segment with a short pause. In VCM-only mode, a missing task slot is rejected; `--enable-asr` permits one clarification turn. |
| `MESSAGE` | Google Chat incoming webhook + optional Moonshine | In VCM-only mode, randomly selects one of at most five configured generic texts and posts it without a recipient label. With `--enable-asr`, Alfred asks for the recipient and message, then posts `To:` and `Message:` lines in its configured Chat space. |
| `PLAY_MUSIC`, `PAUSE`, `STOP`, `NEXT` | Wayne Manor Spotify | Sends acknowledged commands to the laptop browser's Wayne Manor turntable. `PLAY_MUSIC` starts a randomly selected playable track from Wayne Manor's configured playlist. |
| `TIMER`, `ALARM` | systemd user timers | Schedules a one-shot local alert process. Timer durations are bounded; alarm times resolve to the next local occurrence. |
| `CALL` | Wayne Manor telephone or Linphone SIP | Uses the configured backend. Wayne Manor visibly rings its simulated telephone for a bounded interval; Linphone dials the one configured SIP URI through `linphonecsh`. |
| `LIGHT_ON`, `LIGHT_OFF` | WayneManor API | Changes the configured virtual light group's power state. |
| `BRIGHTNESS`, `COLOR` | WayneManor API | Applies the normalized slot. Missing or invalid brightness uses 65%; missing or unsupported color selects a seeded pseudo-random supported color without immediate repetition. |
| `TEMPERATURE` | WayneManor API | Applies a safe setpoint; missing or out-of-range values use 24°C. |

The fallback values are application decisions, not corrected VCM predictions.

### Startup API availability sweep

On every normal startup, Alfred performs non-mutating probes for Wayne Manor,
Open-Meteo, and Google Chat connectivity. Each result is stored in memory for
that process. If a service was unavailable at startup, its actions fail
immediately with a spoken unavailable response instead of waiting for another
network timeout. In particular, an unavailable Wayne Manor endpoint no longer
delays the listening cue while Alfred tries to pause Spotify.

The probe timeout is `runtime.api_probe_timeout_seconds` (one second by
default). Google Chat is checked without posting a message, so this confirms
host connectivity and webhook configuration but cannot prove that the webhook
credential is accepted. Availability is intentionally not polled in the
background; after fixing a service or changing networks, run `alfred restart`
to refresh the flags. Benchmark mode skips the sweep because it disables all
actions.
VCM slot evaluation must continue to score raw model output before these
fallbacks so the end-to-end robustness mechanism cannot inflate slot metrics.

With the Wayne Manor music backend, volume commands control the Spotify
Web Playback SDK on the laptop and do not alter Alfred's microphone/speaker
device. Alfred still uses `osascript`, PipeWire `wpctl`, or ALSA `amixer` to
initialize its own cue/TTS output. Commands are passed directly to the process
API, never through a shell.
Weather, Linphone SIP, and Google Chat messaging need network access; Wayne Manor
uses only its configured local or LAN endpoint.
The WayneManor API is unauthenticated by design and remains an independently
operated service. Start it before exercising its five intents. Alfred uses the
`ALFRED_WAYNE_MANOR_URL` value from private `.env` when present, otherwise it
falls back to `http://127.0.0.1:8765/api/v1`. Use a Pi-accessible LAN URL rather
than loopback when the API runs on another computer. Alfred sends a
unique request ID, uses an eight-second timeout by default, and speaks either the
backend's confirmed result or a safe failure message. The browser receives
live changes from Wayne Manor; Alfred does not open or maintain a WebSocket.

The `TIME` intent deliberately reads the operating system's local timezone. On
a Pi intended for Quezon City, confirm it with `timedatectl` and set it once if
needed with `sudo timedatectl set-timezone Asia/Manila`.

Timers and alarms use `systemd-run --user` with one-second requested accuracy,
`Persistent=true`, and a fixed seven-day maximum timer duration. At expiry a
separate process plays the completion cue and speaks the alert three times.
Transient systemd timers survive an Alfred process restart but may be lost if
the entire user manager is recreated during a reboot; reboot recovery remains
a limitation of this initial scheduler.

The `CALL` intent remains slotless. `actions.call.backend` defaults to
`wayne_manor`, which sends `POST /api/v1/telephone/calls` using Alfred's normal
request ID and lets the simulator stop the ring automatically. Select the real
SIP path with `--call-backend linphone`; it always dials the single
`ALFRED_LINPHONE_DESTINATION` URI. Alfred confirms that Linphone accepted the
dial command, not that the recipient answered. These are Stage 3 choices and do
not add a VCM slot or alter CALL evaluation.

### Optional ASR clarification

The default runtime does not use ASR. When Alfred is launched with
`--enable-asr`, the VCM remains the primary source of all six canonical slots.
After a confident intent prediction, Alfred asks one focused follow-up when the
required slot is missing, invalid, or below the configurable slot-confidence
gate. This applies only to `CREATE_REMINDER`, `TIMER`, `ALARM`, `TEMPERATURE`,
`BRIGHTNESS`, and `COLOR`. `MESSAGE` remains intentionally slotless.

Weather has a separate `--weather-asr` / `--no-weather-asr` switch, so location
transcription can be enabled without enabling ASR clarification for other
intents.

`MESSAGE` has a separate application-level two-turn flow after intent
classification: Alfred asks whom to address, then asks what to send. These
transcripts are passed directly to the action as a validated payload and never
added to the VCM schema or reported as slot predictions. If either turn fails,
Alfred says that nothing was sent and does not call the webhook.

Alfred pauses Spotify for the entire interaction, speaks an intent-specific
question, plays the listening cue, captures one answer with Silero VAD, and
transcribes it locally with quantized Moonshine v2 Tiny English through
Sherpa-ONNX. The answer is validated against the canonical slot type and action
safety range. Cleanup removes terminal punctuation artifacts and immediate
decoder repetitions, but never substitutes one word for another. Invalid or
missing answers cancel the action; Alfred never falls back to a device value
silently.

The model is deliberately excluded from Git. It is not required for the
default VCM-only runtime. Install it from the `alfred` directory using Python
3.12 only when optional ASR is wanted:

```bash
python3.12 tools/install_clarification_asr.py
```

The installer downloads the fixed official Sherpa-ONNX release archive, shows
progress, extracts only `encoder_model.ort`, `decoder_model_merged.ort`,
`tokens.txt`, and available license documentation, and records SHA-256 hashes.
To install an archive copied from another machine instead:

```bash
python3.12 tools/install_clarification_asr.py \
  --archive /path/to/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2
```

Use `--overwrite` only when intentionally replacing an existing installation.
The recognizer loads lazily on the first clarification, so commands with valid,
confident slots do not pay its initialization cost. This is an application
robustness mechanism, not part of the VCM; report VCM slot metrics separately
from end-to-end clarification success, invocation rate, latency, and memory.
The current `0.8` slot-confidence gate is an explicit conservative deployment
setting, not a claim of posterior calibration. Refit it on held-out slot data
before treating clarification-rate comparisons as research conclusions.

### Wayne Manor Spotify

Spotify audio plays from the laptop browser, and Alfred waits for Wayne Manor
to acknowledge each command. See `wayne_manor/README.md` for the one-time
Spotify PKCE setup. The Pi stores no Spotify credential and runs no Spotify
playback service.

When either wake word is detected, Alfred asks Wayne Manor to pause active
playback before recording the command. It restores playback after feedback for
non-media commands when music had been playing. `PAUSE` and `STOP` cancel that
restoration, while `NEXT` advances playback and preserves the pre-interaction
playing state.

For macOS development, deterministic local test doubles remain available for
Spotify, calls, Google Chat messaging, timers, and alarms:

```bash
python -m alfred --mock
```

In this mode Spotify, calls, Google Chat messages, timers, and alarms are mocked.
Open-Meteo, Wayne Manor, SQLite reminders, system volume, audio, TTS, wake-word
detection, and the configured VCM remain real. Combine `--mock` with the VCM
diagnostic flags only when a deterministic prediction is also required:

```bash
python -m alfred --mock --vcm-mode mock --mock-intent PLAY_MUSIC --once
```

Use the Raspberry Pi as the full real-service integration host. After copying
and initializing Alfred, start the full stack with:

```bash
alfred run
alfred status
```

For foreground debugging, stop the Alfred service and start Alfred from its
environment:

```bash
systemctl --user stop alfred.service
cd "$HOME/alfred"
.venv/bin/python -m alfred
```

This foreground Pi process uses every real backend. Stop it with `Ctrl-C`, then
use `alfred run` to return to service-managed operation. Wayne Manor is
intentionally operated outside Alfred and must already be reachable through
`ALFRED_WAYNE_MANOR_URL` before testing its actions.

Keep the incoming webhook URL in the private `.env` file. Validate the host
and path locally without sending a message:

```bash
cp .env.example .env
chmod 600 .env
python tools/test_google_chat.py check
```

For an explicit live Google Chat message test, add the confirmation flag:

```bash
python tools/test_google_chat.py message --text "Alfred test message." --confirm-live
python tools/test_google_chat.py message --recipient Dad --text "I will be home at six." --confirm-live
```

An explicitly supplied process environment variable overrides the `.env`
value. Alfred parses `.env` as data and never executes it as shell code. The
Recipient and message content are captured after the VCM predicts the slotless
`MESSAGE` intent; they are application action data, not canonical VCM slots.
When ASR is disabled, the five `actions.google_chat.fallback_messages` provide
the bounded, recipient-free VCM-only behavior. The random selector avoids
immediately repeating the previous text. Network and API-response failures
become safe spoken results without logging or speaking the credential.
Incoming webhooks always post into their configured Chat space.

## Audio feedback

The committed WAV files are generated reproducibly by
`tools/generate_tones.py`. The operational feedback cues use modal xylophone
synthesis with a mallet transient, wooden-bar overtones, natural damping, and
subtle early reflections. The Easter-egg sounds remain simple tones. Alfred
plays `listening_start` after wake-word detection and, when necessary, after
Spotify confirms its asynchronous pause; there is no
separate wake-detected sound. Regenerate the assets from the `alfred` directory
with:

```bash
python tools/generate_tones.py
```

Action announcements are printed as deterministic text before execution and
spoken through the packaged offline Piper VITS voice. After execution, both
real and mock handlers speak their result. Expected API, database, and system
failures are also spoken; a failure earcon is used when TTS is disabled or
fails. Mock handlers speak an explicit mock result so testing never implies
that a real side effect occurred. The TTS model is loaded once at
startup and uses two CPU threads by default. The configured speed is `1.3`,
which makes the Alan voice moderately faster while preserving intelligibility. A
success earcon is used only when spoken feedback is disabled or fails, avoiding
a redundant high-pitched tone after speech. Configure TTS in
`config/default.json` under `tts`.

## Raspberry Pi service

The complete manual installation, copy, initialization, and verification
procedure is in [`deploy/RASPBERRY_PI_SETUP.md`](deploy/RASPBERRY_PI_SETUP.md).
It includes direct source references for Raspberry Pi OS, ReSpeaker Lite,
native packages, `uv`, optional Linphone, Google Chat messaging, and systemd.

The private `.env` file must contain:

```dotenv
ALFRED_WAYNE_MANOR_URL="http://wayne-manor-host:8765/api/v1"
ALFRED_LINPHONE_DESTINATION="sip:recipient@sip.linphone.org"
ALFRED_GOOGLE_CHAT_WEBHOOK_URL="https://chat.googleapis.com/v1/spaces/.../messages?key=...&token=..."
```

Set the Wayne Manor value to a URL reachable from the Pi. Remove that line only
when the independently operated API really listens on this same Pi at
`127.0.0.1:8765`.

Keep the webhook URL private in `.env`; do not paste it into logs, screenshots,
or committed files.

From the development machine, copy only Alfred's runtime files and models:

```bash
./deploy/copy_to_pi.sh USER@RASPBERRY_PI_HOST
```

After the Pi has been initialized once, deploy later updates with:

```bash
./deploy/sync_to_pi.sh USER@RASPBERRY_PI_HOST
```

Unlike the first-install copy/initialization workflow, this preserves the
Pi's `.env`, virtual environment, and reminders while refreshing locked
dependencies, configuration, service files, and the running Alfred service.

This excludes datasets, experiment outputs, tests, the local virtual
environment, caches, and the development SQLite database. It
includes the private `.env` over encrypted SSH and changes it to owner-only
permissions on the Pi.

After completing the guide's manual software installation and Python setup,
initialize the reminder database, verify the models and audio, install the
Alfred user-service definition and the `alfred` control command, then start the
service:

```bash
cd ~/alfred
./deploy/initialize_pi.sh
```

After initialization, operate the complete application as one unit:

```bash
alfred run
alfred stop
alfred restart
alfred status
alfred logs
```

These commands control Alfred. They deliberately leave PipeWire and WirePlumber
running because those are shared system audio services.

Run the initializer as the normal Alfred user, not as root. It installs no
software and performs no downloads. It uses `sudo` only for the configured
timezone and user-service linger. `Asia/Manila` is the default timezone.
Override it when needed:

```bash
ALFRED_TIMEZONE=Asia/Manila ./deploy/initialize_pi.sh
```

The equivalent manual service commands are:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/alfred.service ~/.config/systemd/user/alfred.service
systemctl --user daemon-reload
systemctl --user enable --now alfred.service
journalctl --user -u alfred.service -f
```

Wake and VAD thresholds must be calibrated on the actual Pi, microphone, room,
and speakers. Do not tune them on the final acceptance recordings.

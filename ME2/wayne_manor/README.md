# Wayne Manor

Wayne Manor is Alfred's independent virtual smart-home environment. A FastAPI
backend owns the simulated light, thermostat, telephone, and Spotify command state; a Three.js frontend
renders that state as a nighttime, bird's-eye manor living room. Its fireplace
provides persistent low ambient light, while the API-controlled chandelier and
wall sconces visibly illuminate the room. Alfred talks only to the HTTP API, so
the simulator can run on the same machine or another machine on the local
network.

The laptop browser can also become a Spotify Connect device named **Wayne Manor
Turntable**. OAuth uses Authorization Code with PKCE entirely in the browser:
the public client ID is configuration, while no client secret or user token is
sent to Alfred or stored by the FastAPI server. The record rotates only when
Spotify reports active playback; it is a state indicator, not a synchronized
audio visualizer.

Climate control is represented as discreet ducted HVAC: a brass wall register
and traditional thermostat dial rather than a modern split-type unit. Changing
the setpoint moves the dial, changes its blue-to-amber color, and briefly
intensifies visible airflow from the register. It also plays the supplied
`frontend/public/assets/air-conditioner.wav` cue at the configured 50% browser
volume.

The simulator implements the existing canonical intents `LIGHT_ON`,
`LIGHT_OFF`, `BRIGHTNESS`, `COLOR`, `TEMPERATURE`, and a demonstration backend
for the existing slotless `CALL` intent. It does not add or alter VCM labels or
slots. A procedural desk telephone shakes and lights while ringing, then stops
after the configured bounded interval.

## Requirements

- Python 3.12
- `uv`
- Node.js 20.19+ or 22.12+ to rebuild the frontend

The browser is the only component that runs Three.js. Alfred and the FastAPI
process do not render the scene.

## Install and run

From the `wayne_manor` directory:

```bash
uv sync --frozen --extra dev
cd frontend
npm ci
npm run build
cd ..
uv run python -m wayne_manor
```

Open <http://127.0.0.1:8765>. Interactive API documentation is available at
<http://127.0.0.1:8765/docs>.

### Spotify setup

1. In the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard),
   open the application with client ID `3fcb96d2b6a84fbdbffff892713e83ca`.
2. Add the exact redirect URI `http://127.0.0.1:8765/` in **Settings**. The
   trailing slash is significant.
3. Ensure the account opening Wayne Manor has Spotify Premium. In development
   mode, add any other tester to the app's user allowlist.
4. Run the production build as above and open exactly
   <http://127.0.0.1:8765/> on the laptop.
5. Select **Connect Spotify**, authorize once, then select **Start turntable**.

The browser must remain open while Alfred controls it. Browser autoplay rules
require the explicit button click. Tokens are held in that browser's local
storage and refreshed with PKCE; clearing site data disconnects the account.
The configured playlist is started from a random track, with shuffle enabled.
`VOLUME_UP` and `VOLUME_DOWN` change this browser player's volume in five-point
steps. The Turntable panel also exposes an exact 0–100% volume slider.

The default `display` mode is the normal demonstration/runtime view. It is
read-only with respect to devices: manual lighting and temperature inputs are
hidden, while light status and the temperature number and gauge remain visible.
The scene can still be orbited and zoomed in either mode. To expose the manual
device-testing controls, start the same application with:

```bash
uv run python -m wayne_manor --ui-mode controls
```

The mode comes from the server and cannot be enabled with a browser query
parameter.

Wayne Manor binds to loopback by default. To let an Alfred process on another
device reach it, copy `config/default.json`, change `server.host` to the Wayne
Manor host's LAN address (or `0.0.0.0` on a trusted local network), and start it
with:

```bash
uv run python -m wayne_manor --config /path/to/local-config.json
```

Then set `actions.wayne_manor.base_url` in Alfred's `config/default.json` to that
host's `/api/v1` URL. Do not expose this unauthenticated simulation API to the
public internet.

## Development

Run the backend:

```bash
uv run python -m wayne_manor --ui-mode controls
```

In a second terminal, run Vite with API and WebSocket proxying:

```bash
cd frontend
npm run dev
```

The production build is served directly by FastAPI, avoiding CORS and a
second runtime service.

## API examples

```bash
curl http://127.0.0.1:8765/api/v1/state

curl -X PUT http://127.0.0.1:8765/api/v1/light-groups/living-room/power \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: manual-light-on' \
  -d '{"on":true}'

curl -X PUT http://127.0.0.1:8765/api/v1/light-groups/living-room/brightness \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: manual-brightness' \
  -d '{"percent":65}'

curl -X PUT http://127.0.0.1:8765/api/v1/light-groups/living-room/color \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: manual-color' \
  -d '{"color":"blue"}'

curl -X PUT http://127.0.0.1:8765/api/v1/thermostats/living-room-climate/setpoint \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: manual-temperature' \
  -d '{"degrees":22,"unit":"celsius"}'

curl -X POST http://127.0.0.1:8765/api/v1/telephone/calls \
  -H 'X-Request-ID: manual-call'

curl -X DELETE http://127.0.0.1:8765/api/v1/telephone/calls/current \
  -H 'X-Request-ID: manual-stop'

# Requires an open, connected Wayne Manor browser. This request waits for its
# acknowledgement instead of reporting success before Spotify actually pauses.
curl -X POST http://127.0.0.1:8765/api/v1/media/pause \
  -H 'X-Request-ID: manual-media-pause'
```

The user-supplied ringtone and air-conditioning cue are bundled at
`frontend/public/assets/telephone-ring.wav` and configured through
`telephone.ringtone_url`, and `frontend/public/assets/air-conditioner.wav` and
configured through the default thermostat. The browser exposes an **Enable room
audio** button because browser autoplay policy normally requires one user
interaction before either sound may start.

Device state is intentionally in memory and resets to `config/default.json` on
restart. Repeating a request with the current value is successful and leaves
the state unchanged. The frontend receives complete snapshots on
`/api/v1/events` and reconnects with bounded backoff if needed.
Spotify credentials are an exception: the browser retains its PKCE tokens in
local storage, but the backend never receives or persists them.

## Verification

```bash
uv run ruff check src tests
uv run pytest -q
cd frontend
npm test
npm run build
```

With the server running, measure API overhead separately from browser
rendering:

```bash
uv run python tools/benchmark_api.py --requests 200
```

On the initial macOS development check, 100 loopback state reads measured
0.431 ms median and 0.667 ms p95. These are development-machine results, not
Raspberry Pi claims. Measure browser frame rate and API latency again on the
actual Pi/display combination before reporting target-device performance.

The backend tests cover configuration, validation, idempotency, concurrency,
error envelopes, and WebSocket propagation. Frontend tests cover snapshot
validation, Spotify state transitions, and thermostat visualization bounds. Alfred's own tests cover the
API client, slot-to-payload mapping, error handling, and action dispatch.

## Raspberry Pi 5

Wayne Manor does not need to run on the Pi. The recommended demonstration setup
is to run Alfred on the Pi and open Wayne Manor on a laptop, with both on the
same trusted LAN. If both run on the Pi, the API and WebSocket are lightweight;
the browser's Three.js rendering is the material graphics workload. The scene
uses procedural low-poly geometry, no shadows, capped pixel density, and no
downloaded assets. The simulated fireplace remains lit when the living-room
lights are off, keeping the nighttime scene legible without making it look like
a daylight interior.

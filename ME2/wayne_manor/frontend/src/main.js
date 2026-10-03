import { api, connectStateSocket } from "./api.js";
import { createWayneManorScene } from "./scene.js";
import { SpotifyTurntable } from "./spotify.js";
import { temperaturePosition, validateWorldState } from "./state.js";
import "./styles.css";

const elements = {
  brightness: document.querySelector("#brightness"),
  brightnessValue: document.querySelector("#brightness-value"),
  colors: document.querySelector("#color-options"),
  connectionDot: document.querySelector("#connection-dot"),
  connectionStatus: document.querySelector("#connection-status"),
  controlPanel: document.querySelector(".control-panel"),
  displayBrightness: document.querySelector("#display-brightness"),
  displayColor: document.querySelector("#display-color"),
  message: document.querySelector("#message"),
  powerOff: document.querySelector("#power-off"),
  powerOn: document.querySelector("#power-on"),
  powerState: document.querySelector("#power-state"),
  temperatureFill: document.querySelector("#temperature-fill"),
  temperatureForm: document.querySelector("#temperature-form"),
  temperatureInput: document.querySelector("#temperature-input"),
  temperatureMaximum: document.querySelector("#temperature-maximum"),
  temperatureMinimum: document.querySelector("#temperature-minimum"),
  temperatureUnit: document.querySelector("#temperature-unit"),
  temperatureValue: document.querySelector("#temperature-value"),
  telephoneName: document.querySelector("#telephone-name"),
  telephoneRing: document.querySelector("#telephone-ring"),
  telephoneState: document.querySelector("#telephone-state"),
  telephoneStop: document.querySelector("#telephone-stop"),
  ringtoneEnable: document.querySelector("#ringtone-enable"),
  spotifyConnect: document.querySelector("#spotify-connect"),
  spotifyNext: document.querySelector("#spotify-next"),
  spotifyPause: document.querySelector("#spotify-pause"),
  spotifyPlay: document.querySelector("#spotify-play"),
  spotifyState: document.querySelector("#spotify-state"),
  spotifyStop: document.querySelector("#spotify-stop"),
  spotifyTrack: document.querySelector("#spotify-track"),
  spotifyVolume: document.querySelector("#spotify-volume"),
  spotifyVolumeValue: document.querySelector("#spotify-volume-value"),
};

const scene = createWayneManorScene(document.querySelector("#scene"));
let capabilities = null;
let world = null;
let requestInProgress = false;
let ringtone = null;
let climateSound = null;
let roomAudioEnabled = false;
let lastTemperatureSetpoint = null;
let spotify = null;
let spotifyHeartbeat = null;
let mediaStatus = null;
let lastMediaCommandId = null;
let lastStatusHeartbeatError = null;
let mediaStatusQueue = Promise.resolve();

function unitSymbol(unit) {
  return unit === "fahrenheit" ? "°F" : "°C";
}

function setMessage(message, error = false) {
  elements.message.textContent = message;
  elements.message.classList.toggle("error", error);
}

function setConnection(status) {
  const labels = {
    connected: "Live",
    connecting: "Connecting",
    reconnecting: "Reconnecting",
    invalid: "Invalid update",
  };
  elements.connectionStatus.textContent = labels[status] || "Offline";
  elements.connectionDot.dataset.status = status;
}

function renderState(incoming) {
  world = validateWorldState(incoming);
  scene.applyWorldState(world);

  const light = world.lights[world.default_light_group];
  elements.powerState.textContent = light.on ? "On" : "Off";
  elements.powerState.classList.toggle("off", !light.on);
  if (document.activeElement !== elements.brightness) {
    elements.brightness.value = String(light.brightness_percent);
  }
  elements.brightnessValue.value = `${Math.round(light.brightness_percent)}%`;
  elements.displayBrightness.value = `${Math.round(light.brightness_percent)}%`;
  elements.displayColor.style.background = light.color_hex;
  elements.displayColor.setAttribute("aria-label", light.color);
  for (const button of elements.colors.querySelectorAll("button")) {
    button.classList.toggle("selected", button.dataset.color === light.color);
    button.setAttribute("aria-pressed", String(button.dataset.color === light.color));
  }

  const thermostat = world.thermostats[world.default_thermostat];
  const currentSetpoint = Number(thermostat.setpoint);
  const temperatureChanged =
    lastTemperatureSetpoint !== null && currentSetpoint !== lastTemperatureSetpoint;
  lastTemperatureSetpoint = currentSetpoint;
  const symbol = unitSymbol(thermostat.unit);
  elements.temperatureValue.textContent = currentSetpoint.toLocaleString();
  elements.temperatureUnit.textContent = symbol;
  elements.temperatureMinimum.textContent = `${thermostat.minimum_setpoint}°`;
  elements.temperatureMaximum.textContent = `${thermostat.maximum_setpoint}°`;
  elements.temperatureFill.style.width = `${temperaturePosition(thermostat) * 100}%`;
  elements.temperatureInput.min = String(thermostat.minimum_setpoint);
  elements.temperatureInput.max = String(thermostat.maximum_setpoint);
  if (document.activeElement !== elements.temperatureInput) {
    elements.temperatureInput.value = String(thermostat.setpoint);
  }
  if (temperatureChanged && climateSound !== null && roomAudioEnabled) {
    climateSound.currentTime = 0;
    climateSound.play().catch(() => {
      roomAudioEnabled = false;
      elements.ringtoneEnable.classList.remove("hidden");
      setMessage("The browser blocked room audio. Enable it again to hear effects.", true);
    });
  }

  const ringing = Boolean(world.telephone?.ringing);
  elements.telephoneState.textContent = ringing ? "Ringing" : "Idle";
  elements.telephoneState.classList.toggle("off", !ringing);
  elements.telephoneState.classList.toggle("ringing", ringing);
  if (ringtone !== null) {
    if (ringing && roomAudioEnabled) {
      ringtone.play().catch(() => {
        roomAudioEnabled = false;
        elements.ringtoneEnable.classList.remove("hidden");
      });
    } else if (!ringing) {
      ringtone.pause();
      ringtone.currentTime = 0;
    }
  }

  const media = world.media || {};
  elements.spotifyState.textContent = media.connected ? media.playback_state : "Offline";
  elements.spotifyState.classList.toggle("off", !media.connected);
  elements.spotifyTrack.textContent = media.track_name
    ? `${media.track_name}${media.artist_name ? ` — ${media.artist_name}` : ""}`
    : media.connected
      ? "Ready for a command"
      : "Spotify is not connected";
  const volumePercent = Math.round(Number(media.volume_percent ?? 50));
  if (document.activeElement !== elements.spotifyVolume) {
    elements.spotifyVolume.value = String(volumePercent);
  }
  elements.spotifyVolumeValue.value = `${volumePercent}%`;
  const command = media.command;
  if (command?.id && command.id !== lastMediaCommandId) {
    lastMediaCommandId = command.id;
    handleMediaCommand(command);
  }
}

async function refreshState() {
  renderState(await api.state());
}

async function mutate(operation) {
  if (requestInProgress) return;
  requestInProgress = true;
  document.body.classList.add("busy");
  try {
    const result = await operation();
    setMessage(result.message);
    await refreshState();
  } catch (error) {
    setMessage(error.message || "The request failed.", true);
  } finally {
    requestInProgress = false;
    document.body.classList.remove("busy");
  }
}

function buildColorOptions() {
  const group = capabilities.lights[capabilities.default_light_group];
  for (const [name, color] of Object.entries(group.supported_colors)) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "color-button";
    button.dataset.color = name;
    button.title = name;
    button.setAttribute("aria-label", `Set lights to ${name}`);
    button.style.setProperty("--swatch", color);
    button.addEventListener("click", () =>
      mutate(() => api.setColor(capabilities.default_light_group, name)),
    );
    elements.colors.append(button);
  }
}

function bindControls() {
  elements.powerOn.addEventListener("click", () =>
    mutate(() => api.setPower(capabilities.default_light_group, true)),
  );
  elements.powerOff.addEventListener("click", () =>
    mutate(() => api.setPower(capabilities.default_light_group, false)),
  );
  elements.brightness.addEventListener("input", () => {
    elements.brightnessValue.value = `${elements.brightness.value}%`;
  });
  elements.brightness.addEventListener("change", () =>
    mutate(() =>
      api.setBrightness(capabilities.default_light_group, Number(elements.brightness.value)),
    ),
  );
  elements.temperatureForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const thermostat = capabilities.thermostats[capabilities.default_thermostat];
    mutate(() =>
      api.setTemperature(
        capabilities.default_thermostat,
        Number(elements.temperatureInput.value),
        thermostat.unit,
      ),
    );
  });
  elements.telephoneRing.addEventListener("click", () => mutate(() => api.ringTelephone()));
  elements.telephoneStop.addEventListener("click", () => mutate(() => api.stopTelephone()));
  elements.spotifyPlay.addEventListener("click", () => mutate(() => api.mediaCommand("play")));
  elements.spotifyPause.addEventListener("click", () => mutate(() => api.mediaCommand("pause")));
  elements.spotifyNext.addEventListener("click", () => mutate(() => api.mediaCommand("next")));
  elements.spotifyStop.addEventListener("click", () => mutate(() => api.mediaCommand("stop")));
}

function normalizedMediaStatus(status, commandId = null, error = null) {
  const boundedInteger = (value) => {
    const number = Number(value);
    return Number.isFinite(number) ? Math.max(0, Math.round(number)) : 0;
  };
  const volume = Number(status?.volume_percent ?? 50);
  return {
    connected: Boolean(status?.connected),
    playback_state: ["STOPPED", "PLAYING", "PAUSED"].includes(status?.playback_state)
      ? status.playback_state
      : "STOPPED",
    device_id: status?.device_id || null,
    track_name: status?.track_name || null,
    artist_name: status?.artist_name || null,
    position_ms: boundedInteger(status?.position_ms),
    duration_ms: boundedInteger(status?.duration_ms),
    volume_percent: Number.isFinite(volume) ? Math.max(0, Math.min(100, volume)) : 50,
    error,
    command_id: commandId,
  };
}

async function publishMediaStatus(status, commandId = null, error = null) {
  mediaStatus = normalizedMediaStatus(status, commandId, error);
  const snapshot = { ...mediaStatus };
  const update = () => api.setMediaStatus(snapshot);
  const result = mediaStatusQueue.then(update, update);
  // Preserve request order so a delayed heartbeat cannot overwrite a newer
  // command acknowledgement. The caller still receives and handles errors.
  mediaStatusQueue = result.catch(() => {});
  await result;
}

async function handleMediaCommand(command) {
  try {
    if (spotify === null) throw new Error("Spotify is not initialized in this browser.");
    const status = await spotify.execute(command.action, command.arguments || {});
    await publishMediaStatus(status, command.id);
  } catch (error) {
    const message = error.message || "Spotify could not complete that command.";
    setMessage(message, true);
    try {
      await publishMediaStatus(spotify?.lastStatus, command.id, message);
    } catch (reportError) {
      setMessage(reportError.message || message, true);
    }
  }
}

async function configureSpotify() {
  spotify = new SpotifyTurntable(capabilities.spotify, {
    onStatus: (status) => {
      mediaStatus = normalizedMediaStatus(status);
      publishMediaStatus(status).catch((error) => setMessage(error.message, true));
    },
    onError: (error) => setMessage(error.message || "Spotify player error.", true),
  });
  const authorized = await spotify.prepare();
  elements.spotifyConnect.textContent = authorized ? "Start turntable" : "Connect Spotify";
  elements.spotifyConnect.addEventListener("click", async () => {
    try {
      elements.spotifyConnect.disabled = true;
      const connected = await spotify.connect();
      if (connected) {
        elements.spotifyConnect.textContent = "Spotify connected";
        elements.spotifyConnect.classList.add("connected");
        await publishMediaStatus(spotify.lastStatus);
      }
    } catch (error) {
      setMessage(error.message || "Spotify could not connect.", true);
    } finally {
      elements.spotifyConnect.disabled = false;
    }
  });
  elements.spotifyVolume.addEventListener("input", () => {
    elements.spotifyVolumeValue.value = `${elements.spotifyVolume.value}%`;
  });
  elements.spotifyVolume.addEventListener("change", () =>
    mutate(() => api.setMediaVolume(Number(elements.spotifyVolume.value))),
  );
  if (spotifyHeartbeat !== null) window.clearInterval(spotifyHeartbeat);
  spotifyHeartbeat = window.setInterval(() => {
    if (mediaStatus !== null) {
      Promise.resolve(spotify?.syncFromPlayer())
        .then((status) => {
          if (status !== undefined) mediaStatus = normalizedMediaStatus(status);
          return publishMediaStatus(mediaStatus, null, null);
        })
        .then(() => {
          lastStatusHeartbeatError = null;
        })
        .catch((error) => {
          const message = error.message || "Spotify status could not be synchronized.";
          if (message !== lastStatusHeartbeatError) setMessage(message, true);
          lastStatusHeartbeatError = message;
        });
    }
  }, 5_000);
}

function configureRoomAudio() {
  const telephone = capabilities.telephone;
  elements.telephoneName.textContent = telephone.display_name;
  if (telephone.ringtone_url) {
    ringtone = new Audio(telephone.ringtone_url);
    ringtone.loop = true;
    ringtone.preload = "auto";
  }
  const thermostat = capabilities.thermostats[capabilities.default_thermostat];
  if (thermostat.change_sound_url) {
    climateSound = new Audio(thermostat.change_sound_url);
    climateSound.preload = "auto";
    climateSound.volume = thermostat.change_sound_volume;
  }
  if (ringtone === null && climateSound === null) return;
  elements.ringtoneEnable.classList.remove("hidden");
  elements.ringtoneEnable.addEventListener("click", async () => {
    try {
      // Unlock each audio element inside the trusted click event so later
      // WebSocket-triggered telephone and climate cues are permitted.
      for (const audio of [ringtone, climateSound].filter(Boolean)) {
        const configuredVolume = audio.volume;
        audio.volume = 0;
        await audio.play();
        audio.pause();
        audio.currentTime = 0;
        audio.volume = configuredVolume;
      }
      roomAudioEnabled = true;
      elements.ringtoneEnable.classList.add("hidden");
      if (world?.telephone?.ringing && ringtone !== null) await ringtone.play();
    } catch {
      roomAudioEnabled = false;
      elements.ringtoneEnable.classList.remove("hidden");
      setMessage("The browser blocked room audio. Try enabling it again.", true);
    }
  });
}

async function start() {
  try {
    const [uiConfig, loadedCapabilities, initialState] = await Promise.all([
      api.uiConfig(),
      api.capabilities(),
      api.state(),
    ]);
    if (!["display", "controls"].includes(uiConfig.mode)) {
      throw new TypeError("WayneManor API returned an invalid UI mode.");
    }
    capabilities = loadedCapabilities;
    configureRoomAudio();
    await configureSpotify();
    document.body.classList.toggle("mode-display", uiConfig.mode === "display");
    document.body.classList.toggle("mode-controls", uiConfig.mode === "controls");
    elements.controlPanel.setAttribute(
      "aria-label",
      uiConfig.mode === "controls" ? "Wayne Manor controls" : "Wayne Manor status",
    );
    if (uiConfig.mode === "controls") {
      buildColorOptions();
      bindControls();
    }
    renderState(initialState);
    setMessage("Wayne Manor systems are ready.");
  } catch (error) {
    setMessage(error.message || "Wayne Manor could not initialize.", true);
    setConnection("offline");
  }

  connectStateSocket({
    onEvent: renderState,
    onStatus: setConnection,
  });
}

start();

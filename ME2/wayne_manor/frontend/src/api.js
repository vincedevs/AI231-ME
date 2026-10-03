export class WayneManorApiError extends Error {
  constructor(message, code = "request_failed", options = undefined) {
    super(message, options);
    this.name = "WayneManorApiError";
    this.code = code;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`/api/v1${path}`, {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...options.headers,
      },
    });
  } catch (error) {
    throw new WayneManorApiError("WayneManor API is unavailable.", "unavailable", { cause: error });
  }

  let document;
  try {
    document = await response.json();
  } catch {
    throw new WayneManorApiError("WayneManor API returned an invalid response.", "invalid_response");
  }
  if (!response.ok) {
    const detail = document?.error;
    throw new WayneManorApiError(detail?.message || "WayneManor API rejected the request.", detail?.code);
  }
  return document;
}

function put(path, body) {
  return request(path, { method: "PUT", body: JSON.stringify(body) });
}

function post(path, body = undefined) {
  return request(path, {
    method: "POST",
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
}

export const api = {
  uiConfig: () => request("/ui-config"),
  capabilities: () => request("/capabilities"),
  state: () => request("/state"),
  setPower: (groupId, on) => put(`/light-groups/${encodeURIComponent(groupId)}/power`, { on }),
  setBrightness: (groupId, percent) =>
    put(`/light-groups/${encodeURIComponent(groupId)}/brightness`, { percent }),
  setColor: (groupId, color) =>
    put(`/light-groups/${encodeURIComponent(groupId)}/color`, { color }),
  setTemperature: (thermostatId, degrees, unit) =>
    put(`/thermostats/${encodeURIComponent(thermostatId)}/setpoint`, { degrees, unit }),
  ringTelephone: () => post("/telephone/calls"),
  stopTelephone: () => request("/telephone/calls/current", { method: "DELETE" }),
  media: () => request("/media"),
  mediaCommand: (action) => post(`/media/${encodeURIComponent(action)}`),
  setMediaVolume: (percent) => put("/media/volume", { percent }),
  setMediaStatus: (status) => post("/media/status", status),
};

export function connectStateSocket({ onEvent, onStatus }) {
  let socket = null;
  let closed = false;
  let retryMilliseconds = 1_000;
  let retryTimer = null;

  function connect() {
    if (closed) return;
    onStatus("connecting");
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(`${protocol}//${window.location.host}/api/v1/events`);
    socket.addEventListener("open", () => {
      retryMilliseconds = 1_000;
      onStatus("connected");
    });
    socket.addEventListener("message", (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === "state.snapshot" || message.type === "state.changed") {
          onEvent(message.state);
        }
      } catch {
        onStatus("invalid");
      }
    });
    socket.addEventListener("close", () => {
      if (closed) return;
      onStatus("reconnecting");
      retryTimer = window.setTimeout(connect, retryMilliseconds);
      retryMilliseconds = Math.min(retryMilliseconds * 2, 8_000);
    });
    socket.addEventListener("error", () => socket.close());
  }

  connect();
  return () => {
    closed = true;
    if (retryTimer !== null) window.clearTimeout(retryTimer);
    socket?.close();
  };
}

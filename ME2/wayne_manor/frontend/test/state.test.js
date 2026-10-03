import assert from "node:assert/strict";
import test from "node:test";

import { temperaturePosition, validateWorldState } from "../src/state.js";
import { api, WayneManorApiError } from "../src/api.js";
import { authorizationEntryUrl, challengeFor } from "../src/spotify.js";

test("state snapshots are checked without exposing revision counters", () => {
  assert.deepEqual(validateWorldState({ lights: {} }), { lights: {} });
  assert.throws(() => validateWorldState(null), /invalid state snapshot/);
  assert.throws(
    () => validateWorldState({ media: { connected: true, playback_state: "playing" } }),
    /invalid Spotify state/,
  );
});

test("temperature position is bounded", () => {
  assert.equal(
    temperaturePosition({ setpoint: 23, minimum_setpoint: 16, maximum_setpoint: 30 }),
    0.5,
  );
  assert.equal(
    temperaturePosition({ setpoint: 50, minimum_setpoint: 16, maximum_setpoint: 30 }),
    1,
  );
});

test("manual controls use the public device API contract", async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (path, options) => {
    calls.push([path, options.method, JSON.parse(options.body)]);
    return { ok: true, json: async () => ({ status: "success" }) };
  };
  try {
    await api.setPower("living-room", false);
    await api.setBrightness("living-room", 45);
    await api.setColor("living-room", "warm white");
    await api.setTemperature("living-room-climate", 22, "celsius");
  } finally {
    globalThis.fetch = originalFetch;
  }
  assert.deepEqual(calls, [
    ["/api/v1/light-groups/living-room/power", "PUT", { on: false }],
    ["/api/v1/light-groups/living-room/brightness", "PUT", { percent: 45 }],
    ["/api/v1/light-groups/living-room/color", "PUT", { color: "warm white" }],
    [
      "/api/v1/thermostats/living-room-climate/setpoint",
      "PUT",
      { degrees: 22, unit: "celsius" },
    ],
  ]);
});

test("telephone controls use the local call endpoints", async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (path, options) => {
    calls.push([path, options.method]);
    return { ok: true, json: async () => ({ status: "success" }) };
  };
  try {
    await api.ringTelephone();
    await api.stopTelephone();
  } finally {
    globalThis.fetch = originalFetch;
  }
  assert.deepEqual(calls, [
    ["/api/v1/telephone/calls", "POST"],
    ["/api/v1/telephone/calls/current", "DELETE"],
  ]);
});

test("media controls use the command and browser-status endpoints", async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (path, options) => {
    calls.push([path, options.method, options.body ? JSON.parse(options.body) : null]);
    return { ok: true, status: 200, json: async () => ({ status: "success" }) };
  };
  try {
    await api.mediaCommand("play");
    await api.setMediaVolume(65);
    await api.setMediaStatus({ connected: true, playback_state: "PLAYING" });
  } finally {
    globalThis.fetch = originalFetch;
  }
  assert.deepEqual(calls, [
    ["/api/v1/media/play", "POST", null],
    ["/api/v1/media/volume", "PUT", { percent: 65 }],
    ["/api/v1/media/status", "POST", { connected: true, playback_state: "PLAYING" }],
  ]);
});

test("network errors preserve their original cause", async () => {
  const originalFetch = globalThis.fetch;
  const networkFailure = new Error("connection refused");
  globalThis.fetch = async () => {
    throw networkFailure;
  };
  try {
    await assert.rejects(api.state(), (error) => {
      assert.ok(error instanceof WayneManorApiError);
      assert.equal(error.code, "unavailable");
      assert.equal(error.cause, networkFailure);
      return true;
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("UI mode is loaded from the server contract", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (path) => {
    assert.equal(path, "/api/v1/ui-config");
    return { ok: true, json: async () => ({ mode: "display" }) };
  };
  try {
    assert.deepEqual(await api.uiConfig(), { mode: "display" });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("PKCE challenge remains correct without Web Crypto subtle", async () => {
  const verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
  assert.equal(
    await challengeFor(verifier, null),
    "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
  );
});

test("Spotify authorization enters the registered origin before creating PKCE state", () => {
  assert.equal(
    authorizationEntryUrl("http://192.168.1.9:8765", "http://127.0.0.1:8765/"),
    "http://127.0.0.1:8765/?spotify_authorize=1",
  );
  assert.equal(
    authorizationEntryUrl("http://127.0.0.1:8765", "http://127.0.0.1:8765/"),
    null,
  );
});

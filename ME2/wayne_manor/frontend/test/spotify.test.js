import assert from "node:assert/strict";
import test from "node:test";

import { SpotifyTurntable } from "../src/spotify.js";

const tracks = [
  {
    type: "track",
    uri: "spotify:track:trackA",
    name: "Track A",
    artists: [{ name: "Artist A" }],
  },
  {
    type: "track",
    uri: "spotify:track:trackB",
    name: "Track B",
    artists: [{ name: "Artist B" }],
  },
];

function stateFor(track, { paused = false, position = 0 } = {}) {
  return track === null
    ? null
    : {
        paused,
        position,
        duration: 180_000,
        track_window: { current_track: track },
      };
}

function makeTurntable(initialState) {
  globalThis.window = { setTimeout, clearTimeout };
  let currentState = initialState;
  const turntable = new SpotifyTurntable(
    {
      client_id: "test-client",
      default_uri: "spotify:playlist:playlistA",
      initial_volume: 0.3,
    },
    { onStatus() {}, onError(error) { throw error; } },
  );
  turntable.connected = true;
  turntable.deviceId = "test-device";
  turntable.player = {
    async getCurrentState() {
      return currentState;
    },
    async pause() {
      if (currentState) currentState = { ...currentState, paused: true };
    },
    async resume() {
      if (currentState) currentState = { ...currentState, paused: false };
    },
    async seek(position) {
      if (currentState) currentState = { ...currentState, position };
    },
    async setVolume() {},
  };
  turntable.spotifyApi = async (path, method, body) => {
    if (path.startsWith("/playlists/")) {
      assert.match(path, /^\/playlists\/playlistA\/items\?/);
      return {
        total: 4,
        items: [
          { item: tracks[0] },
          { item: { ...tracks[1], is_playable: false } },
          // Retain tolerance for a legacy response shape.
          { track: tracks[1] },
          { track: null },
        ],
      };
    }
    if (path.startsWith("/me/player/play") && method === "PUT") {
      const track = tracks.find((item) => item.uri === body.uris[0]);
      currentState = stateFor(track);
      return null;
    }
    throw new Error(`Unexpected Spotify API request: ${path}`);
  };
  return turntable;
}

test("play from STOPPED chooses a random playable playlist track", async () => {
  const originalRandom = Math.random;
  Math.random = () => 0.75;
  try {
    const turntable = makeTurntable(null);
    const status = await turntable.execute("play");
    assert.equal(status.playback_state, "PLAYING");
    assert.equal(status.track_name, "Track B");
    assert.equal(status.position_ms, 0);
  } finally {
    Math.random = originalRandom;
  }
});

test("play from PAUSED resumes the same track at the same position", async () => {
  const turntable = makeTurntable(stateFor(tracks[0], { paused: true, position: 42_500 }));
  const status = await turntable.execute("play");
  assert.equal(status.playback_state, "PLAYING");
  assert.equal(status.track_name, "Track A");
  assert.equal(status.position_ms, 42_500);
});

test("pause is idempotent and preserves the current playback position", async () => {
  const turntable = makeTurntable(stateFor(tracks[0], { position: 23_000 }));
  const first = await turntable.execute("pause");
  const second = await turntable.execute("pause");
  assert.equal(first.playback_state, "PAUSED");
  assert.equal(second.playback_state, "PAUSED");
  assert.equal(second.position_ms, 23_000);
});

test("stop resets playback and the next play starts a fresh random track", async () => {
  const originalRandom = Math.random;
  Math.random = () => 0;
  try {
    const turntable = makeTurntable(stateFor(tracks[0], { position: 57_000 }));
    // The SDK may acknowledge seek(0) before its next state snapshot reflects it.
    turntable.player.seek = async () => {};
    const stopped = await turntable.execute("stop");
    assert.equal(stopped.playback_state, "STOPPED");
    assert.equal(stopped.track_name, null);
    assert.equal(stopped.position_ms, 0);
    assert.equal((await turntable.syncFromPlayer()).playback_state, "STOPPED");
    assert.equal((await turntable.execute("stop")).playback_state, "STOPPED");
    const restarted = await turntable.execute("play");
    assert.equal(restarted.playback_state, "PLAYING");
    assert.equal(restarted.track_name, "Track A");
  } finally {
    Math.random = originalRandom;
  }
});

test("skip advances using playable playlist entries and retains paused state", async () => {
  const turntable = makeTurntable(stateFor(tracks[0], { paused: true, position: 40_000 }));
  const status = await turntable.execute("next");
  assert.equal(status.playback_state, "PAUSED");
  assert.equal(status.track_name, "Track B");
  assert.equal(status.position_ms, 0);
});

test("skip waits for a delayed Spotify player-state update", async () => {
  const turntable = makeTurntable(stateFor(tracks[0], { position: 35_000 }));
  let currentState = stateFor(tracks[0], { position: 35_000 });
  turntable.player.getCurrentState = async () => currentState;
  turntable.spotifyApi = async (path, method, body) => {
    if (path.startsWith("/playlists/")) {
      return {
        total: 2,
        items: [{ item: tracks[0] }, { item: tracks[1] }],
      };
    }
    if (path.startsWith("/me/player/play") && method === "PUT") {
      const nextTrack = tracks.find((item) => item.uri === body.uris[0]);
      setTimeout(() => { currentState = stateFor(nextTrack); }, 500);
      return null;
    }
    throw new Error(`Unexpected Spotify API request: ${path}`);
  };

  const status = await turntable.execute("next");
  assert.equal(status.playback_state, "PLAYING");
  assert.equal(status.track_name, "Track B");
});

test("malformed and empty playlist responses fail explicitly", async () => {
  const malformed = makeTurntable(null);
  malformed.spotifyApi = async () => ({ items: [], total: "unknown" });
  await assert.rejects(malformed.loadPlayableTracks(), /invalid playlist track count/);

  const empty = makeTurntable(null);
  empty.spotifyApi = async () => ({ total: 0, items: [] });
  await assert.rejects(empty.loadPlayableTracks(), /no playable tracks/);
});

test("Spotify API retries an expired access token once without exposing it", async () => {
  const turntable = makeTurntable(null);
  turntable.spotifyApi = SpotifyTurntable.prototype.spotifyApi.bind(turntable);
  const tokenRequests = [];
  turntable.accessToken = async (forceRefresh) => {
    tokenRequests.push(forceRefresh);
    return "opaque-test-access-token";
  };
  const originalFetch = globalThis.fetch;
  let requests = 0;
  globalThis.fetch = async () => {
    requests += 1;
    return requests === 1
      ? { status: 401, ok: false, text: async () => "" }
      : { status: 200, ok: true, text: async () => JSON.stringify({ devices: [] }) };
  };
  try {
    assert.deepEqual(await turntable.spotifyApi("/me/player/devices"), { devices: [] });
    assert.deepEqual(tokenRequests, [false, true]);
    assert.equal(requests, 2);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("Spotify rejects malformed JSON and sanitizes request network failures", async () => {
  const turntable = makeTurntable(null);
  turntable.spotifyApi = SpotifyTurntable.prototype.spotifyApi.bind(turntable);
  turntable.accessToken = async () => "opaque-test-access-token";
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => ({ status: 200, ok: true, text: async () => "LS2opaque" });
    await assert.rejects(turntable.spotifyApi("/me/player/devices"), /invalid API response/);

    globalThis.fetch = async () => {
      throw new Error("request failed with opaque-test-access-token");
    };
    await assert.rejects(
      turntable.spotifyApi("/me/player/devices"),
      (error) => error.message === "Spotify is unavailable. Check the network connection.",
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

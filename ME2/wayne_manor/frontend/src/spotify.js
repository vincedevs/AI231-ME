const ACCESS_TOKEN_KEY = "wayne-manor.spotify.v2.access-token";
const REFRESH_TOKEN_KEY = "wayne-manor.spotify.v2.refresh-token";
const EXPIRES_AT_KEY = "wayne-manor.spotify.v2.expires-at";
const LEGACY_TOKEN_KEY = "wayne-manor.spotify.tokens";
const VERIFIER_KEY = "wayne-manor.spotify.verifier";
const STATE_KEY = "wayne-manor.spotify.state";
const AUTHORIZE_URL = "https://accounts.spotify.com/authorize";
const TOKEN_URL = "https://accounts.spotify.com/api/token";
const API_URL = "https://api.spotify.com/v1";
const AUTHORIZATION_ENTRY_PARAMETER = "spotify_authorize";

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function readableSpotifyError(document, fallback) {
  const description = typeof document?.error_description === "string"
    ? document.error_description
    : typeof document?.error?.message === "string"
      ? document.error.message
      : null;
  return description || fallback;
}

export function authorizationEntryUrl(currentOrigin, redirectUri) {
  const redirect = new URL(redirectUri);
  if (currentOrigin === redirect.origin) return null;
  redirect.searchParams.set(AUTHORIZATION_ENTRY_PARAMETER, "1");
  return redirect.toString();
}

function base64Url(bytes) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replaceAll("=", "");
}

function randomText(byteCount = 48) {
  const bytes = new Uint8Array(byteCount);
  if (typeof globalThis.crypto?.getRandomValues !== "function") {
    throw new Error("This browser cannot securely authorize Spotify.");
  }
  globalThis.crypto.getRandomValues(bytes);
  return base64Url(bytes);
}

function rotateRight(value, amount) {
  return (value >>> amount) | (value << (32 - amount));
}

export function sha256Fallback(input) {
  const bytes = new TextEncoder().encode(input);
  const paddedLength = Math.ceil((bytes.length + 9) / 64) * 64;
  const padded = new Uint8Array(paddedLength);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  const view = new DataView(padded.buffer);
  const bitLength = bytes.length * 8;
  view.setUint32(paddedLength - 8, Math.floor(bitLength / 2 ** 32), false);
  view.setUint32(paddedLength - 4, bitLength >>> 0, false);

  const constants = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  ];
  const hash = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]);
  const words = new Uint32Array(64);
  for (let offset = 0; offset < paddedLength; offset += 64) {
    for (let index = 0; index < 16; index += 1) {
      words[index] = view.getUint32(offset + index * 4, false);
    }
    for (let index = 16; index < 64; index += 1) {
      const first = words[index - 15];
      const second = words[index - 2];
      const sigma0 = rotateRight(first, 7) ^ rotateRight(first, 18) ^ (first >>> 3);
      const sigma1 = rotateRight(second, 17) ^ rotateRight(second, 19) ^ (second >>> 10);
      words[index] = (words[index - 16] + sigma0 + words[index - 7] + sigma1) >>> 0;
    }
    let [a, b, c, d, e, f, g, h] = hash;
    for (let index = 0; index < 64; index += 1) {
      const sum1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
      const choice = (e & f) ^ (~e & g);
      const temporary1 = (h + sum1 + choice + constants[index] + words[index]) >>> 0;
      const sum0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
      const majority = (a & b) ^ (a & c) ^ (b & c);
      const temporary2 = (sum0 + majority) >>> 0;
      h = g;
      g = f;
      f = e;
      e = (d + temporary1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temporary1 + temporary2) >>> 0;
    }
    hash[0] = (hash[0] + a) >>> 0;
    hash[1] = (hash[1] + b) >>> 0;
    hash[2] = (hash[2] + c) >>> 0;
    hash[3] = (hash[3] + d) >>> 0;
    hash[4] = (hash[4] + e) >>> 0;
    hash[5] = (hash[5] + f) >>> 0;
    hash[6] = (hash[6] + g) >>> 0;
    hash[7] = (hash[7] + h) >>> 0;
  }
  const output = new Uint8Array(32);
  const outputView = new DataView(output.buffer);
  hash.forEach((value, index) => outputView.setUint32(index * 4, value, false));
  return output;
}

export async function challengeFor(verifier, subtle = globalThis.crypto?.subtle) {
  const encoded = new TextEncoder().encode(verifier);
  const digest = subtle
    ? await subtle.digest("SHA-256", encoded)
    : sha256Fallback(verifier);
  return base64Url(new Uint8Array(digest));
}

async function tokenRequest(body) {
  let response;
  try {
    response = await fetch(TOKEN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams(body),
    });
  } catch {
    throw new Error("Spotify authorization service is unavailable.");
  }
  let payload;
  try {
    payload = await response.text();
  } catch {
    throw new Error("Spotify's authorization service returned an unreadable response.");
  }
  let document;
  try {
    document = JSON.parse(payload);
  } catch {
    throw new Error("Spotify's authorization service returned an invalid response.");
  }
  if (!isRecord(document)) {
    throw new Error("Spotify's authorization service returned an invalid response.");
  }
  if (!response.ok || typeof document.access_token !== "string" || !document.access_token) {
    throw new Error(readableSpotifyError(document, "Spotify authorization failed."));
  }
  if (document.refresh_token !== undefined && typeof document.refresh_token !== "string") {
    throw new Error("Spotify authorization returned an invalid refresh token.");
  }
  if (document.expires_in !== undefined && !Number.isFinite(Number(document.expires_in))) {
    throw new Error("Spotify authorization returned an invalid token lifetime.");
  }
  return document;
}

function loadSpotifySdk() {
  if (window.Spotify?.Player) return Promise.resolve();
  return new Promise((resolve, reject) => {
    window.onSpotifyWebPlaybackSDKReady = resolve;
    const existing = document.querySelector('script[src="https://sdk.scdn.co/spotify-player.js"]');
    const timeout = window.setTimeout(
      () => reject(new Error("Spotify's player did not load. Check the network and try again.")),
      15_000,
    );
    const ready = () => {
      window.clearTimeout(timeout);
      if (!window.Spotify?.Player) {
        reject(new Error("Spotify loaded an invalid player SDK."));
        return;
      }
      resolve();
    };
    window.onSpotifyWebPlaybackSDKReady = ready;
    if (existing) {
      existing.addEventListener("error", () => {
        window.clearTimeout(timeout);
        reject(new Error("Spotify's player could not load."));
      }, { once: true });
      return;
    }
    const script = document.createElement("script");
    script.src = "https://sdk.scdn.co/spotify-player.js";
    script.async = true;
    script.addEventListener("error", () => {
      window.clearTimeout(timeout);
      reject(new Error("Spotify's player could not load."));
    }, { once: true });
    document.head.append(script);
  });
}

export class SpotifyTurntable {
  constructor(config, { onStatus, onError }) {
    this.config = config;
    this.onStatus = onStatus;
    this.onError = onError;
    this.player = null;
    this.deviceId = null;
    this.connected = false;
    this.playbackState = "STOPPED";
    this.currentTrackUri = null;
    this.playlistTracks = null;
    this.stoppedTrackUri = null;
    this.commandQueue = Promise.resolve();
    this.readyWaiter = null;
    this.volumePercent = Math.round(this.config.initial_volume * 100);
    this.lastStatus = this.emptyStatus();
  }

  emptyStatus() {
    return {
      connected: this.connected,
      playback_state: this.playbackState,
      device_id: this.deviceId,
      track_name: this.playbackState === "STOPPED" ? null : this.lastStatus?.track_name || null,
      artist_name: this.playbackState === "STOPPED" ? null : this.lastStatus?.artist_name || null,
      position_ms: 0,
      duration_ms: 0,
      volume_percent: this.volumePercent,
      error: null,
    };
  }

  publish(status) {
    const previous = this.lastStatus;
    this.lastStatus = { ...status, connected: this.connected, device_id: this.deviceId };
    const significantChange = [
      "connected",
      "playback_state",
      "device_id",
      "track_name",
      "artist_name",
      "volume_percent",
      "error",
    ].some((key) => previous?.[key] !== this.lastStatus[key]);
    if (significantChange) this.onStatus(this.lastStatus);
  }

  hasAuthorization() {
    return Boolean(localStorage.getItem(ACCESS_TOKEN_KEY));
  }

  async authorize() {
    const entryUrl = authorizationEntryUrl(window.location.origin, this.config.redirect_uri);
    if (entryUrl !== null) {
      // sessionStorage is isolated by origin. Enter the registered loopback
      // origin before creating the PKCE transaction so Spotify's callback can
      // retrieve the same verifier and anti-forgery state.
      window.location.assign(entryUrl);
      return;
    }
    const verifier = randomText();
    const state = randomText(24);
    sessionStorage.setItem(VERIFIER_KEY, verifier);
    sessionStorage.setItem(STATE_KEY, state);
    const parameters = new URLSearchParams({
      client_id: this.config.client_id,
      response_type: "code",
      redirect_uri: this.config.redirect_uri,
      code_challenge_method: "S256",
      code_challenge: await challengeFor(verifier),
      state,
      scope: this.config.scopes.join(" "),
    });
    window.location.assign(`${AUTHORIZE_URL}?${parameters}`);
  }

  async prepare() {
    const parameters = new URLSearchParams(window.location.search);
    if (parameters.get(AUTHORIZATION_ENTRY_PARAMETER) === "1") {
      history.replaceState({}, "", window.location.pathname);
      await this.authorize();
      return false;
    }
    if (parameters.has("error")) {
      history.replaceState({}, "", window.location.pathname);
      throw new Error(`Spotify authorization was declined: ${parameters.get("error")}.`);
    }
    const code = parameters.get("code");
    if (code !== null) {
      const expectedState = sessionStorage.getItem(STATE_KEY);
      const verifier = sessionStorage.getItem(VERIFIER_KEY);
      if (!expectedState || parameters.get("state") !== expectedState || !verifier) {
        sessionStorage.removeItem(STATE_KEY);
        sessionStorage.removeItem(VERIFIER_KEY);
        history.replaceState({}, "", window.location.pathname);
        this.onError(
          new Error("Spotify authorization expired. Click Connect Spotify and try again."),
        );
        return false;
      }
      let tokens;
      try {
        tokens = await tokenRequest({
        client_id: this.config.client_id,
        grant_type: "authorization_code",
        code,
        redirect_uri: this.config.redirect_uri,
        code_verifier: verifier,
        });
      } catch (error) {
        sessionStorage.removeItem(STATE_KEY);
        sessionStorage.removeItem(VERIFIER_KEY);
        history.replaceState({}, "", window.location.pathname);
        throw error;
      }
      this.storeTokens(tokens);
      sessionStorage.removeItem(STATE_KEY);
      sessionStorage.removeItem(VERIFIER_KEY);
      history.replaceState({}, "", window.location.pathname);
    }
    if (!this.hasAuthorization()) return false;
    await loadSpotifySdk();
    this.createPlayer();
    return true;
  }

  storeTokens(tokens) {
    const previous = this.readTokens();
    localStorage.removeItem(LEGACY_TOKEN_KEY);
    localStorage.setItem(ACCESS_TOKEN_KEY, String(tokens.access_token).trim());
    const refreshToken = tokens.refresh_token || previous?.refresh_token;
    if (refreshToken) localStorage.setItem(REFRESH_TOKEN_KEY, String(refreshToken).trim());
    localStorage.setItem(
      EXPIRES_AT_KEY,
      String(Date.now() + Number(tokens.expires_in || 3600) * 1000),
    );
  }

  readTokens() {
    // Discard the pre-v2 aggregate value: an interrupted earlier login could
    // leave an opaque credential where JSON was expected.
    localStorage.removeItem(LEGACY_TOKEN_KEY);
    const accessToken = localStorage.getItem(ACCESS_TOKEN_KEY);
    if (!accessToken) return null;
    return {
      access_token: accessToken,
      refresh_token: localStorage.getItem(REFRESH_TOKEN_KEY),
      expires_at: Number(localStorage.getItem(EXPIRES_AT_KEY) || 0),
    };
  }

  clearTokens() {
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
    localStorage.removeItem(EXPIRES_AT_KEY);
    localStorage.removeItem(LEGACY_TOKEN_KEY);
  }

  async accessToken(forceRefresh = false) {
    const tokens = this.readTokens();
    if (!tokens?.access_token) throw new Error("Spotify is not authorized.");
    if (!forceRefresh && Date.now() < Number(tokens.expires_at) - 60_000) {
      return tokens.access_token;
    }
    if (!tokens.refresh_token) {
      this.clearTokens();
      throw new Error("Spotify must be connected again.");
    }
    let refreshed;
    try {
      refreshed = await tokenRequest({
        client_id: this.config.client_id,
        grant_type: "refresh_token",
        refresh_token: tokens.refresh_token,
      });
    } catch {
      this.clearTokens();
      throw new Error("Spotify authorization expired. Connect Spotify again.");
    }
    this.storeTokens(refreshed);
    return refreshed.access_token;
  }

  createPlayer() {
    if (this.player !== null) return;
    this.player = new window.Spotify.Player({
      name: this.config.device_name,
      volume: this.config.initial_volume,
      enableMediaSession: true,
      getOAuthToken: (callback) => {
        this.accessToken().then(callback).catch((error) => this.onError(error));
      },
    });
    this.player.addListener("ready", (readyEvent) => {
      const deviceId = isRecord(readyEvent) ? readyEvent.device_id : null;
      if (typeof deviceId !== "string" || !deviceId) {
        this.readyWaiter?.reject(new Error("Spotify did not provide an active Wayne Manor device."));
        return;
      }
      this.connected = true;
      this.deviceId = deviceId;
      this.readyWaiter?.resolve(deviceId);
      this.readyWaiter = null;
      this.syncFromPlayer().catch((error) => this.onError(error));
      this.publish(this.lastStatus);
    });
    this.player.addListener("not_ready", () => {
      this.connected = false;
      this.deviceId = null;
      this.publish(this.emptyStatus());
    });
    this.player.addListener("player_state_changed", (state) => this.observeState(state));
    for (const event of [
      "initialization_error",
      "authentication_error",
      "account_error",
      "playback_error",
    ]) {
      this.player.addListener(event, (eventData) => {
        const message = isRecord(eventData) ? eventData.message : null;
        const safeMessage = typeof message === "string" && message.trim()
          ? message.slice(0, 300)
          : "Spotify reported an unspecified player error.";
        const error = new Error(`Spotify ${event.replaceAll("_", " ")}: ${safeMessage}`);
        if (event === "authentication_error" || event === "account_error") {
          this.connected = false;
          this.deviceId = null;
        }
        this.onError(error);
      });
    }
  }

  observeState(state) {
    if (state === null) {
      this.playbackState = "STOPPED";
      this.currentTrackUri = null;
      this.publish(this.emptyStatus());
      return;
    }
    if (!isRecord(state)) {
      throw new Error("Spotify returned an invalid playback state.");
    }
    const track = isRecord(state.track_window?.current_track)
      ? state.track_window.current_track
      : null;
    const uri = typeof track?.uri === "string" && track.uri.startsWith("spotify:track:")
      ? track.uri
      : null;
    const position = Number.isFinite(Number(state.position)) ? Math.max(0, Number(state.position)) : 0;
    if (
      this.stoppedTrackUri !== null &&
      uri === this.stoppedTrackUri &&
      state.paused === true
    ) {
      // Spotify has no native stop operation. Once pause + seek(0) succeeds,
      // keep Alfred's STOPPED state even if the SDK reports a stale position.
      this.playbackState = "STOPPED";
      this.currentTrackUri = null;
      this.publish(this.statusFromState(state, null));
      return;
    }
    this.stoppedTrackUri = null;
    this.currentTrackUri = uri;
    if (typeof state.paused !== "boolean") {
      throw new Error("Spotify returned a playback state without a valid pause flag.");
    }
    this.playbackState = !uri ? "STOPPED" : state.paused ? "PAUSED" : "PLAYING";
    this.publish(this.statusFromState(state, track));
  }

  statusFromState(state, track) {
    const artists = Array.isArray(track?.artists)
      ? track.artists.map((artist) => typeof artist?.name === "string" ? artist.name : "").filter(Boolean)
      : [];
    const stopped = this.playbackState === "STOPPED";
    const duration = Number(state.duration);
    return {
      connected: this.connected,
      playback_state: this.playbackState,
      device_id: this.deviceId,
      track_name: stopped ? null : typeof track?.name === "string" ? track.name : null,
      artist_name: stopped ? null : artists.length ? artists.join(", ") : null,
      position_ms: stopped || !Number.isFinite(Number(state.position)) ? 0 : Math.max(0, Math.round(Number(state.position))),
      duration_ms: stopped || !Number.isFinite(duration) ? 0 : Math.max(0, Math.round(duration)),
      volume_percent: this.volumePercent,
      error: null,
    };
  }

  async syncFromPlayer() {
    if (!this.player) return this.lastStatus;
    const state = await this.player.getCurrentState();
    this.observeState(state);
    return this.lastStatus;
  }

  async waitForState(
    predicate,
    { timeoutMs = 8_000, description = "the requested playback state" } = {},
  ) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      const status = await this.syncFromPlayer();
      if (predicate(status)) return status;
      // Spotify's Web Playback SDK updates asynchronously after Web API
      // commands. Poll slowly enough to avoid hammering the SDK, but long
      // enough to tolerate device and track hand-offs on slower browsers.
      await new Promise((resolve) => window.setTimeout(resolve, 200));
    }
    throw new Error(
      `Spotify did not confirm ${description} within ${Math.ceil(timeoutMs / 1_000)} seconds. `
      + `Last reported state: ${this.playbackState}.`,
    );
  }

  async connect() {
    if (!this.hasAuthorization()) {
      await this.authorize();
      return false;
    }
    if (this.player === null) {
      await loadSpotifySdk();
      this.createPlayer();
    }
    await this.player.activateElement();
    if (!this.connected) {
      const readiness = new Promise((resolve, reject) => {
        const timeout = window.setTimeout(() => {
          this.readyWaiter = null;
          reject(new Error("Spotify player did not become ready. Check the active device and Premium account."));
        }, 15_000);
        this.readyWaiter = {
          resolve: (deviceId) => {
            window.clearTimeout(timeout);
            resolve(deviceId);
          },
          reject: (error) => {
            window.clearTimeout(timeout);
            this.readyWaiter = null;
            reject(error);
          },
        };
      });
      try {
        const connected = await this.player.connect();
        if (!connected) throw new Error("Spotify could not connect the Wayne Manor player.");
        await readiness;
      } catch (error) {
        this.readyWaiter?.reject(error);
        await readiness.catch(() => {});
        throw error;
      }
    }
    if (!this.deviceId) throw new Error("Spotify has no active Wayne Manor playback device.");
    return true;
  }

  async spotifyApi(path, method = "GET", body = undefined, retry = true) {
    const token = await this.accessToken(!retry);
    let response;
    try {
      response = await fetch(`${API_URL}${path}`, {
        method,
        headers: {
          Authorization: `Bearer ${token}`,
          ...(body === undefined ? {} : { "Content-Type": "application/json" }),
        },
        ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      });
    } catch {
      throw new Error("Spotify is unavailable. Check the network connection.");
    }
    if (response.status === 401 && retry) return this.spotifyApi(path, method, body, false);
    if (response.status === 204) return null;
    let payload;
    try {
      payload = await response.text();
    } catch {
      throw new Error("Spotify returned an unreadable API response.");
    }
    let document = null;
    if (payload.trim()) {
      try {
        document = JSON.parse(payload);
      } catch {
        throw new Error("Spotify returned an invalid API response.");
      }
      if (!isRecord(document)) throw new Error("Spotify returned an invalid API response.");
    }
    if (!response.ok) {
      throw new Error(readableSpotifyError(document, `Spotify rejected the command (${response.status}).`));
    }
    if (response.status === 202 || !payload.trim()) return null;
    return document;
  }

  async loadPlayableTracks() {
    if (this.playlistTracks) return this.playlistTracks;
    const match = /^spotify:playlist:([A-Za-z0-9]+)$/.exec(String(this.config.default_uri));
    if (!match) throw new Error("The configured Spotify source is not a valid playlist URI.");
    const playlistId = encodeURIComponent(match[1]);
    const tracks = [];
    let offset = 0;
    let total = null;
    while (total === null || offset < total) {
      const page = await this.spotifyApi(
        `/playlists/${playlistId}/items?limit=50&offset=${offset}`,
      );
      if (!isRecord(page) || !Array.isArray(page.items)) {
        throw new Error("Spotify returned an incomplete playlist response.");
      }
      const rawTotal = Number(page.total);
      if (!Number.isInteger(rawTotal) || rawTotal < 0) {
        throw new Error("Spotify returned an invalid playlist track count.");
      }
      total = rawTotal;
      for (const entry of page.items) {
        // Spotify's 2026 playlist API calls entries `item`; accept the former
        // `track` field too for compatible proxies/older fixtures.
        const track = isRecord(entry)
          ? isRecord(entry.item)
            ? entry.item
            : isRecord(entry.track)
              ? entry.track
              : null
          : null;
        if (!track || track.is_local === true || track.is_playable === false) continue;
        if (
          track.type !== "track" ||
          typeof track.uri !== "string" ||
          !/^spotify:track:[A-Za-z0-9]+$/.test(track.uri) ||
          typeof track.name !== "string" ||
          !track.name.trim()
        ) continue;
        tracks.push({
          uri: track.uri,
          name: track.name,
          artists: Array.isArray(track.artists)
            ? track.artists.filter((artist) => isRecord(artist) && typeof artist.name === "string")
            : [],
        });
      }
      if (page.items.length === 0 || offset + page.items.length >= total) break;
      offset += page.items.length;
    }
    if (tracks.length === 0) {
      throw new Error("The configured Spotify playlist has no playable tracks.");
    }
    this.playlistTracks = tracks;
    return tracks;
  }

  async startTrack(track) {
    if (!isRecord(track) || typeof track.uri !== "string") {
      throw new Error("Spotify could not select a valid track.");
    }
    await this.spotifyApi(
      `/me/player/play?device_id=${encodeURIComponent(this.deviceId)}`,
      "PUT",
      { uris: [track.uri], position_ms: 0 },
    );
    this.stoppedTrackUri = null;
    await this.waitForState(
      (status) => status.playback_state === "PLAYING" && this.currentTrackUri === track.uri,
      { timeoutMs: 12_000, description: "the selected track starting" },
    );
    return this.lastStatus;
  }

  async play() {
    const actual = await this.syncFromPlayer();
    if (actual.playback_state === "PLAYING") return actual;
    if (actual.playback_state === "PAUSED" && this.currentTrackUri) {
      this.stoppedTrackUri = null;
      await this.player.resume();
      return this.waitForState(
        (status) => status.playback_state === "PLAYING",
        { description: "playback resuming" },
      );
    }
    const tracks = await this.loadPlayableTracks();
    const selected = tracks[Math.min(tracks.length - 1, Math.floor(Math.random() * tracks.length))];
    return this.startTrack(selected);
  }

  async pause() {
    const actual = await this.syncFromPlayer();
    if (actual.playback_state !== "PLAYING") return actual;
    await this.player.pause();
    return this.waitForState(
      (status) => status.playback_state === "PAUSED",
      { description: "playback pausing" },
    );
  }

  async stop() {
    const actual = await this.syncFromPlayer();
    if (actual.playback_state === "STOPPED") return actual;
    const stoppedUri = this.currentTrackUri;
    if (actual.playback_state === "PLAYING") {
      await this.player.pause();
    }
    await this.player.seek(0);
    this.stoppedTrackUri = stoppedUri;
    this.playbackState = "STOPPED";
    this.currentTrackUri = null;
    this.publish(this.emptyStatus());
    return this.lastStatus;
  }

  async next() {
    const actual = await this.syncFromPlayer();
    if (actual.playback_state === "STOPPED" || !this.currentTrackUri) return this.play();
    const wasPaused = actual.playback_state === "PAUSED";
    const tracks = await this.loadPlayableTracks();
    const currentIndex = tracks.findIndex((track) => track.uri === this.currentTrackUri);
    const nextIndex = currentIndex < 0 ? 0 : (currentIndex + 1) % tracks.length;
    const nextTrack = tracks[nextIndex];
    await this.spotifyApi(
      `/me/player/play?device_id=${encodeURIComponent(this.deviceId)}`,
      "PUT",
      { uris: [nextTrack.uri], position_ms: 0 },
    );
    this.stoppedTrackUri = null;
    await this.waitForState(
      (status) => status.playback_state === "PLAYING" && this.currentTrackUri === nextTrack.uri,
      { timeoutMs: 12_000, description: "the next track starting" },
    );
    if (wasPaused) {
      await this.player.pause();
      await this.waitForState(
        (status) => status.playback_state === "PAUSED",
        { timeoutMs: 5_000, description: "the skipped track remaining paused" },
      );
    }
    return this.lastStatus;
  }

  async execute(action, arguments_ = {}) {
    void arguments_;
    if (!this.connected || !this.deviceId || this.player === null) {
      throw new Error("Spotify is not connected to the Wayne Manor turntable.");
    }
    const run = async () => {
      if (action === "play" || action === "resume") return this.play();
      if (action === "pause") return this.pause();
      if (action === "stop") return this.stop();
      if (action === "next") return this.next();
      if (action === "set_volume") {
        const percent = Number(arguments_.percent);
        if (!Number.isFinite(percent) || percent < 0 || percent > 100) {
          throw new Error("Wayne Manor received an invalid Spotify volume.");
        }
        await this.player.setVolume(percent / 100);
        this.volumePercent = percent;
        await this.syncFromPlayer();
        return this.lastStatus;
      }
      throw new Error(`Unsupported Spotify action: ${action}`);
    };
    const result = this.commandQueue.then(run, run);
    this.commandQueue = result.catch(() => {});
    return result;
  }
}

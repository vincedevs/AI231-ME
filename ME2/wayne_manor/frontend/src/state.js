export function validateWorldState(incoming) {
  if (!incoming || typeof incoming !== "object" || Array.isArray(incoming)) {
    throw new TypeError("Wayne Manor returned an invalid state snapshot");
  }
  if (incoming.media !== undefined) {
    const media = incoming.media;
    if (
      !media ||
      typeof media !== "object" ||
      Array.isArray(media) ||
      typeof media.connected !== "boolean" ||
      !["STOPPED", "PLAYING", "PAUSED"].includes(media.playback_state)
    ) {
      throw new TypeError("Wayne Manor returned an invalid Spotify state");
    }
  }
  return incoming;
}

export function temperaturePosition(thermostat) {
  const span = thermostat.maximum_setpoint - thermostat.minimum_setpoint;
  if (!(span > 0)) {
    return 0;
  }
  return Math.max(0, Math.min(1, (thermostat.setpoint - thermostat.minimum_setpoint) / span));
}

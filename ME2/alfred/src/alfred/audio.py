from __future__ import annotations

import queue
from collections import deque
from types import TracebackType
from typing import Any, Self


class AudioFramer:
    """Reframe arbitrary mono audio blocks without losing boundary samples."""

    def __init__(self, frame_samples: int) -> None:
        if frame_samples <= 0:
            raise ValueError("frame_samples must be positive")
        self.frame_samples = int(frame_samples)
        self._blocks: deque[Any] = deque()
        self._samples = 0

    def clear(self) -> None:
        self._blocks.clear()
        self._samples = 0

    def push(self, block: Any) -> list[Any]:
        import numpy as np

        values = np.asarray(block).reshape(-1)
        if values.size:
            self._blocks.append(values.copy())
            self._samples += int(values.size)
        frames = []
        while self._samples >= self.frame_samples:
            needed = self.frame_samples
            parts = []
            while needed:
                current = self._blocks[0]
                take = min(needed, int(current.size))
                parts.append(current[:take])
                if take == current.size:
                    self._blocks.popleft()
                else:
                    self._blocks[0] = current[take:]
                needed -= take
                self._samples -= take
            frames.append(np.concatenate(parts))
        return frames


def playback_output_device(audio_config: dict[str, Any]) -> int | str | None:
    """Use the OS default sink when configured, preserving system volume control."""
    if bool(audio_config["use_system_default_output"]):
        return None
    return audio_config.get("output_device")


def matching_device_indices(devices: Any, name_fragment: str) -> tuple[int, int]:
    """Select deterministic PortAudio input/output indexes by device-name fragment."""
    fragment = name_fragment.casefold().strip()
    if not fragment:
        raise ValueError("Audio device match cannot be empty")
    inputs = [
        index
        for index, device in enumerate(devices)
        if fragment in str(device["name"]).casefold()
        and int(device.get("max_input_channels", 0)) > 0
    ]
    outputs = [
        index
        for index, device in enumerate(devices)
        if fragment in str(device["name"]).casefold()
        and int(device.get("max_output_channels", 0)) > 0
    ]
    if not inputs or not outputs:
        raise RuntimeError(
            f"No full-duplex audio devices matched {name_fragment!r}; "
            f"input matches={inputs}, output matches={outputs}"
        )
    shared = sorted(set(inputs) & set(outputs))
    if shared:
        return shared[0], shared[0]
    return inputs[0], outputs[0]


def resample_for_output(samples: Any, source_rate: int, target_rate: int) -> Any:
    """Linearly resample mono or channel-last float audio for device playback."""
    import numpy as np

    values = np.asarray(samples, dtype=np.float32)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Audio sample rates must be positive")
    if values.ndim not in {1, 2}:
        raise ValueError("Playback audio must be mono or channel-last stereo")
    if source_rate == target_rate or values.shape[0] == 0:
        return values
    output_length = max(1, round(values.shape[0] * target_rate / source_rate))
    source_positions = np.arange(values.shape[0], dtype=np.float64)
    output_positions = np.linspace(0, values.shape[0] - 1, output_length)
    if values.ndim == 1:
        return np.interp(output_positions, source_positions, values).astype(np.float32)
    channels = [
        np.interp(output_positions, source_positions, values[:, channel])
        for channel in range(values.shape[1])
    ]
    return np.column_stack(channels).astype(np.float32)


def prepare_playback_audio(
    samples: Any,
    source_rate: int,
    target_rate: int,
    gain: float,
    *,
    fade_ms: float = 10.0,
    trailing_silence_ms: float = 30.0,
) -> Any:
    """Apply Alfred's common playback gain, fades, limiter, and resampling.

    The gain is deliberately relative to the operating-system output volume.
    On Raspberry Pi, the initializer starts the system sink at 50%; TTS and
    feedback retain their own smaller gains as the user changes system volume.
    """
    import numpy as np

    if not 0 < gain <= 1:
        raise ValueError("Playback gain must be greater than 0 and at most 1")
    if fade_ms < 0 or trailing_silence_ms < 0:
        raise ValueError("Playback fade and trailing silence cannot be negative")
    audio = np.asarray(samples, dtype=np.float32).copy()
    if audio.ndim not in {1, 2} or not bool(np.isfinite(audio).all()):
        raise ValueError("Playback audio must be finite mono or channel-last stereo")
    audio *= float(gain)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.95:
        audio *= 0.95 / peak
    fade_samples = min(audio.shape[0] // 2, round(source_rate * fade_ms / 1000))
    if fade_samples > 0:
        fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        if audio.ndim == 2:
            fade = fade[:, None]
        audio[:fade_samples] *= fade
        audio[-fade_samples:] *= fade[::-1]
    silence_shape = (round(source_rate * trailing_silence_ms / 1000),) + audio.shape[1:]
    if silence_shape[0] > 0:
        audio = np.concatenate((audio, np.zeros(silence_shape, dtype=np.float32)))
    return resample_for_output(audio, source_rate, target_rate)


class MicrophoneInput:
    """One bounded-queue PortAudio input stream shared by all Alfred stages."""

    def __init__(
        self,
        *,
        sample_rate: int,
        block_samples: int,
        device: int | str | None,
        queue_blocks: int,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.block_samples = int(block_samples)
        self.device = device
        self.queue: queue.Queue[bytes] = queue.Queue(maxsize=int(queue_blocks))
        self.stream: Any | None = None
        self.dropped_blocks = 0

    @staticmethod
    def list_devices() -> Any:
        try:
            import sounddevice as sd
        except ImportError as error:
            raise RuntimeError("sounddevice is required to list audio devices") from error
        return sd.query_devices()

    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        del frames, time_info
        if status:
            self.dropped_blocks += int(bool(getattr(status, "input_overflow", False)))
        payload = bytes(indata)
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            self.dropped_blocks += 1
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(payload)

    def start(self) -> None:
        if self.stream is not None:
            return
        try:
            import sounddevice as sd
        except ImportError as error:
            raise RuntimeError("sounddevice is required for microphone capture") from error
        self.stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_samples,
            device=self.device,
            channels=1,
            dtype="int16",
            callback=self._callback,
        )
        self.stream.start()

    def stop(self) -> None:
        stream = self.stream
        self.stream = None
        if stream is not None:
            stream.stop()
            stream.close()
        self.clear()

    def clear(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                return

    def read(self, timeout: float = 1.0) -> Any:
        import numpy as np

        try:
            payload = self.queue.get(timeout=timeout)
        except queue.Empty as error:
            raise TimeoutError("No microphone audio arrived before the read timeout") from error
        return np.frombuffer(payload, dtype="<i2").copy()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.stop()


def int16_to_float32(samples: Any) -> Any:
    import numpy as np

    return np.asarray(samples, dtype=np.float32).reshape(-1) / 32768.0

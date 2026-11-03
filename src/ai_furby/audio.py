"""Microphone capture and speaker playback. Uses a simple loudness threshold to find where speech starts and ends."""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt, sosfilt_zi

log = logging.getLogger(__name__)

INPUT_SAMPLE_RATE = 16000  # what faster-whisper expects
BLOCK_SECONDS = 0.1
SILENCE_LEVEL = 1e-4  # samples quieter than this count as digital silence
NOISE_FLOOR = 0.002  # faint noise (about -54 dB) used in pauses instead of digital silence
MAX_PEAK = 0.8  # playback headroom so the device does not clip
PLAYBACK_TAIL_SECONDS = 0.4  # quiet padding after each playback so the last word is not clipped
NOISE_WINDOW_BLOCKS = 20  # about 2 s of quiet audio used to estimate room noise
NOISE_MULTIPLIER = 3.0  # speech must be this many times louder than the room noise
MIN_SPEECH_BLOCKS = 3  # 0.3 s of sustained loudness before recording starts
SPEECH_HIGHPASS_HZ = 300.0  # loudness is judged above this; motor rumble and room hum sit below the voice band
MIN_UTTERANCE_SECONDS = 0.4  # shorter bursts are noise (such as the Furby's own movement), not speech to send to Mistral
MIC_QUEUE_BLOCKS = 200  # about 20 s of audio; if nothing reads it, the oldest audio is dropped
MIC_STALL_SECONDS = 3.0  # with no audio for this long, the microphone is reopened


def find_input_device(name_part: str) -> int | None:
    return find_device(name_part, "input")


def find_device(name_part: str, kind: str) -> int | None:
    """Returns the index of an `input` or `output` device whose name contains `name_part`.

    Windows lists the same device under several host APIs, so WASAPI is preferred when it is available.
    """
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    matches = []
    for index, device in enumerate(sd.query_devices()):
        if device[channel_key] > 0 and name_part.lower() in device["name"].lower():
            host_api = sd.query_hostapis(device["hostapi"])["name"]
            matches.append((0 if "WASAPI" in host_api else 1, index))
    return min(matches)[1] if matches else None


ARECORD_RATE = 48000  # the voiceHAT card records at this rate; arecord does not resample for us


def is_alsa_device(device: str | int | None) -> bool:
    """True for names like `plughw:2,0`, which mean an ALSA card to read with arecord."""
    return isinstance(device, str) and device.startswith(("plughw:", "hw:"))


def capture_rate(device: str | int | None) -> int:
    if is_alsa_device(device):
        return ARECORD_RATE
    return int(sd.query_devices(device, kind="input")["default_samplerate"])


def open_microphone(device: str | int | None, rate: int, block: int):
    """A context manager whose `read(block)` returns (samples, overflowed), like sd.InputStream.

    PortAudio can't reach some ALSA cards on the Pi, so those are read through arecord instead.
    """
    if is_alsa_device(device):
        return _ArecordStream(device, rate, block)
    return sd.InputStream(samplerate=rate, channels=1, dtype="float32", blocksize=block, device=device)


class _ArecordStream:
    def __init__(self, device: str, rate: int, block: int) -> None:
        self._device = device
        self._rate = rate
        self._block = block
        self._process: subprocess.Popen | None = None

    def __enter__(self) -> "_ArecordStream":
        command = ["arecord", "-q", "-D", self._device, "-f", "S32_LE", "-r", str(self._rate), "-c", "1", "-t", "raw"]
        self._process = subprocess.Popen(command, stdout=subprocess.PIPE)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process.wait()

    def kill(self) -> None:
        if self._process is not None:
            self._process.kill()

    def read(self, frames: int) -> tuple[np.ndarray, bool]:
        raw = self._process.stdout.read(frames * 4)
        if len(raw) < frames * 4:
            raise RuntimeError(f"arecord stopped on {self._device}; check MIC_DEVICE and that nothing else holds the card")
        return _s32_to_float(raw).reshape(-1, 1), False


def _s32_to_float(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0


class MicFeed:
    """Keeps the microphone open on a background thread, so no audio is lost between the wake word and the question.

    Blocks of mono float32 audio at `rate` queue up until something reads them. `drain()` throws away audio that
    was captured earlier, such as the Furby's own voice during playback.
    """

    def __init__(self, device: str | int | None) -> None:
        self.rate = capture_rate(device)
        self.block = int(self.rate * BLOCK_SECONDS)
        self._device = device
        self._blocks: queue.Queue[np.ndarray] = queue.Queue(maxsize=MIC_QUEUE_BLOCKS)
        self._running = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name="mic-feed", daemon=True)
        self._stream = None
        self._last_block = time.monotonic()

    def start(self) -> None:
        self._running.set()
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        self._thread.join(timeout=2)

    def restart(self) -> None:
        """Closes the microphone and opens it again. Playback can leave the capture stalled, so this runs after each reply."""
        log.info("Reopening the microphone")
        self._running.clear()
        kill = getattr(self._stream, "kill", None)
        if kill is not None:
            kill()  # unblocks a reader that is waiting on a stalled capture process
        self._thread.join(timeout=2)
        self._error = None
        self._blocks = queue.Queue(maxsize=MIC_QUEUE_BLOCKS)  # drop audio captured before the restart
        self._last_block = time.monotonic()
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="mic-feed", daemon=True)
        self._thread.start()

    def read(self, timeout: float) -> np.ndarray:
        """The next block of audio. Raises queue.Empty if none arrives within `timeout` seconds."""
        if self._error is not None:
            raise RuntimeError(f"microphone stopped: {self._error}") from self._error
        try:
            return self._blocks.get(timeout=timeout)
        except queue.Empty:
            if time.monotonic() - self._last_block > MIC_STALL_SECONDS:
                self.restart()  # the capture has stalled; reopen it, then let the caller try again
            raise

    def drain(self) -> None:
        while True:
            try:
                self._blocks.get_nowait()
            except queue.Empty:
                return

    def _run(self) -> None:
        try:
            with open_microphone(self._device, self.rate, self.block) as stream:
                self._stream = stream
                while self._running.is_set():
                    data, _overflowed = stream.read(self.block)
                    if self._blocks.full():
                        self._blocks.get_nowait()
                    self._blocks.put_nowait(data[:, 0].copy())
                    self._last_block = time.monotonic()
        except Exception as error:  # reported to the next reader instead of stopping silently
            self._error = error
            log.exception("Microphone feed stopped")


def record_utterance(
    feed: MicFeed,
    threshold: float = 0.01,
    silence_seconds: float = 1.0,
    max_seconds: float = 15.0,
    wait_seconds: float = 30.0,
    on_speech_start: Callable[[], None] | None = None,
) -> np.ndarray:
    """Waits for speech on `feed`, records until a pause, and returns mono float32 audio at 16 kHz.

    `on_speech_start` runs once, from this thread, when speech is confirmed.
    Returns an empty array if no speech starts within `wait_seconds`.
    """
    device_rate = feed.rate
    silent_blocks_to_stop = int(silence_seconds / BLOCK_SECONDS)
    max_blocks = int(max_seconds / BLOCK_SECONDS)
    wait_blocks = int(wait_seconds / BLOCK_SECONDS)

    captured: list[np.ndarray] = []
    recent_levels: deque[float] = deque(maxlen=NOISE_WINDOW_BLOCKS)
    pending: list[np.ndarray] = []  # loud blocks that may be the start of speech, not yet confirmed
    speaking = False
    silent_run = 0
    speech_filter = butter(2, SPEECH_HIGHPASS_HZ, btype="highpass", fs=device_rate, output="sos")
    filter_state = sosfilt_zi(speech_filter) * 0.0

    for index in range(max_blocks + wait_blocks):
        samples = remove_offset(_next_block(feed))
        filtered, filter_state = sosfilt(speech_filter, samples, zi=filter_state)
        level = float(np.sqrt(np.mean(np.square(filtered))))
        if not speaking:
            recent_levels.append(level)
        # Mic gain changes with Windows settings, so judge speech against the current room noise, not a fixed number.
        loud = level > _adaptive_threshold(threshold, recent_levels)

        if not speaking:
            if index >= wait_blocks:
                break
            if not loud:
                pending.clear()
                continue
            pending.append(samples)
            # Short clicks and bumps are loud for only a block or two; require a sustained burst before starting.
            if len(pending) >= MIN_SPEECH_BLOCKS:
                speaking = True
                captured.extend(pending)
                pending.clear()
                if on_speech_start is not None:
                    on_speech_start()
            continue

        captured.append(samples)
        silent_run = 0 if loud else silent_run + 1
        if silent_run >= silent_blocks_to_stop:
            break

    if not captured:
        return np.zeros(0, dtype=np.float32)
    speech = np.concatenate(captured)
    if speech.size < MIN_UTTERANCE_SECONDS * device_rate:
        return np.zeros(0, dtype=np.float32)  # too short to be speech; nothing to send
    return _resample(speech, device_rate, INPUT_SAMPLE_RATE)


def _next_block(feed: MicFeed) -> np.ndarray:
    while True:
        try:
            return feed.read(timeout=1.0)
        except queue.Empty:
            continue


def remove_offset(samples: np.ndarray) -> np.ndarray:
    """Removes a steady DC offset from a block of audio. Some mics add one, which inflates the measured level."""
    if samples.size == 0:
        return samples
    return samples - float(np.mean(samples))


def _adaptive_threshold(minimum: float, recent_levels: deque[float]) -> float:
    """Speech must be well above the recent quiet level. `minimum` is the floor for very quiet rooms."""
    if len(recent_levels) < NOISE_WINDOW_BLOCKS // 2:
        return minimum
    noise = float(np.median(recent_levels))
    return max(minimum, noise * NOISE_MULTIPLIER)


def _resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate:
        return samples
    target_length = int(len(samples) * to_rate / from_rate)
    old_positions = np.linspace(0.0, 1.0, num=len(samples))
    new_positions = np.linspace(0.0, 1.0, num=target_length)
    return np.interp(new_positions, old_positions, samples).astype(np.float32)


def play(samples: np.ndarray, sample_rate: int, device: str | None = None) -> None:
    """Plays int16 or float32 mono samples on the named output device (or the default) and waits until they finish."""
    output = find_device(device, "output") if device else None
    if device and output is None:
        raise RuntimeError(f"No output device matches {device!r}")
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    # Convert to the output device's own rate ourselves, so Windows doesn't resample in the middle of playback.
    device_rate = int(sd.query_devices(output, kind="output")["default_samplerate"])
    if device_rate != sample_rate:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(device_rate, sample_rate)
        samples = resample_poly(samples, device_rate // divisor, sample_rate // divisor).astype(np.float32)
    # Resampling can overshoot full scale, and the device then clips the peaks. Keep headroom.
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > MAX_PEAK:
        samples = samples * (MAX_PEAK / peak)
    # Bluetooth headsets can power down during digital silence, which clips the start of the next phrase.
    # Keep a faint noise floor in the pauses so the stream never goes fully quiet.
    rng = np.random.default_rng()
    quiet = np.abs(samples) < SILENCE_LEVEL
    samples = np.where(quiet, rng.uniform(-NOISE_FLOOR, NOISE_FLOOR, size=samples.shape).astype(np.float32), samples)
    # Some Bluetooth and ALSA paths drop the last buffer, which clips the final word. A short quiet tail avoids that.
    tail = rng.uniform(-NOISE_FLOOR, NOISE_FLOOR, size=int(device_rate * PLAYBACK_TAIL_SECONDS)).astype(np.float32)
    samples = np.concatenate([samples, tail])
    try:
        _play_once(samples, device_rate, output)
    except sd.PortAudioError:
        # Bluetooth audio devices can appear or change while the program runs, and PortAudio keeps the old list.
        # Refresh the device list once and try again.
        log.info("First playback attempt failed; retrying once")
        sd._terminate()
        sd._initialize()
        output = find_device(device, "output") if device else None
        _play_once(samples, device_rate, output)


def _play_once(samples: np.ndarray, rate: int, output: int | None) -> None:
    log.info("Playing on device %s at %d Hz", output if output is not None else "default", rate)
    sd.play(samples, samplerate=rate, device=output)
    sd.wait()

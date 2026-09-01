"""
Microphone-based real-time audio capture engine.
Uses sounddevice with non-blocking callbacks to stream 16kHz mono audio into thread-safe queues.
"""

import queue
import sys
import numpy as np
from typing import Optional

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

class AudioCapture:
    """
    Continuous real-time audio acquisition engine.
    Streams 16-bit PCM (or float32) mono audio at 16 kHz.
    """

    def __init__(self, sample_rate: int = 16000, block_size: int = 1600, device: Optional[int] = None):
        """
        Initialize AudioCapture.
        :param sample_rate: Sampling frequency in Hz (default 16000)
        :param block_size: Number of samples per chunk (1600 = 100ms)
        :param device: Hardware audio input device index (None for default)
        """
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[object] = None
        self._is_active: bool = False

    def _audio_callback(self, indata, frames, time_info, status):
        """Non-blocking streaming callback invoked by PortAudio."""
        if status:
            print(f"[AudioCapture Warning] Stream status: {status}", file=sys.stderr)

        data = indata[:, 0] if indata.ndim > 1 else indata
        if data.dtype == np.float32:
            data = (data * 32767.0).astype(np.int16)
        self.audio_queue.put(data.copy())

    def start(self):
        """Start non-blocking audio capture stream."""
        if not SOUNDDEVICE_AVAILABLE:
            print("[AudioCapture Error] 'sounddevice' library is not installed. Live microphone disabled.", file=sys.stderr)
            return

        if self._is_active:
            return

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            device=self.device,
            channels=1,
            dtype='int16',
            callback=self._audio_callback
        )
        self._stream.start()
        self._is_active = True

    def stop(self):
        """Stop and close the audio stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._is_active = False

    def get_chunk(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """
        Retrieve next 100ms audio chunk from the buffer queue.
        :param timeout: Maximum wait time in seconds
        :return: 1D numpy int16 array of size (block_size,) or None
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_active(self) -> bool:
        """Return stream active status."""
        return self._is_active

    @staticmethod
    def calculate_rms(chunk: np.ndarray) -> float:
        """Calculate Root Mean Square energy of an audio frame."""
        if chunk is None or len(chunk) == 0:
            return 0.0
        float_chunk = chunk.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(float_chunk ** 2)))

    @staticmethod
    def calculate_db(chunk: np.ndarray) -> float:
        """Calculate decibel level relative to full scale (dBFS)."""
        rms = AudioCapture.calculate_rms(chunk)
        return float(20.0 * np.log10(rms + 1e-6))

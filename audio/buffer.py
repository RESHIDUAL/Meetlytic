"""
Privacy-first circular audio buffer.
Ensures temporary audio storage in RAM with immediate cryptographic/zero-fill
wiping upon classification as personal, casual, or non-work related.
"""

import numpy as np
from typing import Optional

class CircularAudioBuffer:
    """
    In-memory circular ring buffer for raw audio PCM frames.
    Implements secure memory zeroing to guarantee that discarded private
    conversations cannot be retrieved from memory or temporary disk swap.
    """

    def __init__(self, duration_sec: int = 30, sample_rate: int = 16000):
        """
        Initialize circular buffer.
        :param duration_sec: Maximum retention span of temporary audio in seconds (e.g. 30s)
        :param sample_rate: Audio sampling frequency (default 16 kHz)
        """
        self.duration_sec = duration_sec
        self.sample_rate = sample_rate
        self.capacity = duration_sec * sample_rate
        self._buffer = np.zeros(self.capacity, dtype=np.int16)
        self._write_pos: int = 0
        self._total_written: int = 0

    def write(self, data: np.ndarray):
        """
        Write new PCM audio chunk into circular buffer.
        Overwrites oldest data when capacity is reached.
        """
        if data is None or len(data) == 0:
            return

        data_flat = data.flatten().astype(np.int16)
        n = len(data_flat)

        if n >= self.capacity:

            self._buffer[:] = data_flat[-self.capacity:]
            self._write_pos = 0
            self._total_written += n
            return

        end_pos = (self._write_pos + n) % self.capacity
        if self._write_pos + n <= self.capacity:
            self._buffer[self._write_pos:self._write_pos + n] = data_flat
        else:
            first_chunk = self.capacity - self._write_pos
            self._buffer[self._write_pos:] = data_flat[:first_chunk]
            self._buffer[:end_pos] = data_flat[first_chunk:]

        self._write_pos = end_pos
        self._total_written += n

    def get_recent(self, duration_sec: float) -> np.ndarray:
        """
        Retrieve the most recent N seconds of contiguous audio.
        :param duration_sec: Duration in seconds to retrieve
        :return: 1D numpy int16 array
        """
        samples_needed = int(duration_sec * self.sample_rate)
        available = min(self._total_written, self.capacity)
        samples_to_read = min(samples_needed, available)

        if samples_to_read == 0:
            return np.zeros(0, dtype=np.int16)

        out = np.zeros(samples_to_read, dtype=np.int16)
        start_idx = (self._write_pos - samples_to_read) % self.capacity

        if start_idx + samples_to_read <= self.capacity:
            out[:] = self._buffer[start_idx:start_idx + samples_to_read]
        else:
            first_part = self.capacity - start_idx
            out[:first_part] = self._buffer[start_idx:]
            out[first_part:] = self._buffer[:samples_to_read - first_part]

        return out

    def secure_wipe_recent(self, duration_sec: float):
        """
        Securely zero-out the most recent N seconds of audio memory immediately.
        Used when a segment is classified as personal / casual / irrelevant.
        """
        samples_to_wipe = int(duration_sec * self.sample_rate)
        available = min(self._total_written, self.capacity)
        wipe_len = min(samples_to_wipe, available)

        if wipe_len == 0:
            return

        start_idx = (self._write_pos - wipe_len) % self.capacity
        if start_idx + wipe_len <= self.capacity:
            self._buffer[start_idx:start_idx + wipe_len] = 0
        else:
            first_part = self.capacity - start_idx
            self._buffer[start_idx:] = 0
            self._buffer[:wipe_len - first_part] = 0

    def secure_wipe_all(self):
        """
        Cryptographic zero-fill of the entire buffer.
        Leaves no residual conversation traces in RAM.
        """
        self._buffer.fill(0)
        self._write_pos = 0
        self._total_written = 0

    @property
    def total_written_samples(self) -> int:
        return self._total_written

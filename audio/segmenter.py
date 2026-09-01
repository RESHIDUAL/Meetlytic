"""
Speech segmentation module.
Accumulates VAD decisions, merges intra-sentence micro-pauses (< 350ms),
and extracts clean conversational audio segments for processing.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class SpeechSegment:
    """Represents an isolated continuous speech turn."""
    start_sample: int
    end_sample: int
    audio_data: np.ndarray
    duration_sec: float
    timestamp_str: str = ""

class SpeechSegmenter:
    """
    Groups contiguous speech frames into full sentences or conversational phrases.
    """

    def __init__(self, sample_rate: int = 16000,
                 min_duration_ms: int = 250,
                 max_duration_ms: int = 10000,
                 merge_gap_ms: int = 350):
        self.sample_rate = sample_rate
        self.min_samples = int(sample_rate * min_duration_ms / 1000)
        self.max_samples = int(sample_rate * max_duration_ms / 1000)
        self.merge_gap_samples = int(sample_rate * merge_gap_ms / 1000)

        self._active_start_sample: Optional[int] = None
        self._last_speech_sample: Optional[int] = None
        self._accumulated_audio: List[np.ndarray] = []
        self._current_sample_cursor: int = 0
        self._pending_segments: List[SpeechSegment] = []

    def feed_audio_and_vad(self, chunk: np.ndarray, is_speech: bool):
        """
        Feed audio chunk with overall VAD activity flag.
        """
        if chunk is None or len(chunk) == 0:
            return

        n = len(chunk)
        chunk_start = self._current_sample_cursor
        chunk_end = chunk_start + n
        self._current_sample_cursor = chunk_end

        if is_speech:
            if self._active_start_sample is None:
                self._active_start_sample = chunk_start
                self._accumulated_audio = [chunk.copy()]
            else:
                self._accumulated_audio.append(chunk.copy())
            self._last_speech_sample = chunk_end
        else:
            if self._active_start_sample is not None:

                silence_gap = chunk_end - (self._last_speech_sample or chunk_start)
                if silence_gap > self.merge_gap_samples:

                    self._finalize_active_segment()
                else:

                    self._accumulated_audio.append(chunk.copy())

        if self._active_start_sample is not None:
            current_len = sum(len(c) for c in self._accumulated_audio)
            if current_len >= self.max_samples:
                self._finalize_active_segment()

    def _finalize_active_segment(self):
        """Finalize and package accumulated speech segment."""
        if not self._accumulated_audio or self._active_start_sample is None:
            self._active_start_sample = None
            self._accumulated_audio = []
            return

        combined_audio = np.concatenate(self._accumulated_audio)
        duration_samples = len(combined_audio)

        if duration_samples >= self.min_samples:
            seg = SpeechSegment(
                start_sample=self._active_start_sample,
                end_sample=self._active_start_sample + duration_samples,
                audio_data=combined_audio,
                duration_sec=duration_samples / self.sample_rate
            )
            self._pending_segments.append(seg)

        self._active_start_sample = None
        self._last_speech_sample = None
        self._accumulated_audio = []

    def get_completed_segments(self) -> List[SpeechSegment]:
        """Retrieve and clear all completed speech segments ready for feature extraction."""
        segments = self._pending_segments[:]
        self._pending_segments.clear()
        return segments

    def reset(self):
        """Reset segmenter state."""
        self._active_start_sample = None
        self._last_speech_sample = None
        self._accumulated_audio = []
        self._current_sample_cursor = 0
        self._pending_segments.clear()

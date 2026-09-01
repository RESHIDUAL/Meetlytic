"""
Voice Activity Detection (VAD) implemented purely from first-principles mathematical signal processing.
Combines Short-Time Energy (STE), Zero-Crossing Rate (ZCR), and Spectral Flatness (SFM / Wiener Entropy)
with dynamic noise-floor adaptation and an onset/hangover finite state machine.
"""

import numpy as np
from typing import List, Tuple, Optional

class VoiceActivityDetector:
    """
    Mathematical Voice Activity Detector.
    No machine learning or external neural networks.
    """

    def __init__(self, sample_rate: int = 16000, frame_size: int = 400, hop_size: int = 160):
        """
        Initialize VAD.
        :param sample_rate: Sampling frequency (16000 Hz)
        :param frame_size: Frame length in samples (400 = 25ms)
        :param hop_size: Frame shift / step in samples (160 = 10ms)
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size

        self.noise_mean_energy: float = 1e-4
        self.noise_std_energy: float = 1e-5
        self.noise_mean_sfm: float = 0.8
        self.noise_std_sfm: float = 0.1
        self.calibrated: bool = False

        self.energy_threshold: float = 5e-4
        self.sfm_threshold: float = 0.42

        self.state: str = "SILENCE"
        self.onset_counter: int = 0
        self.hangover_counter: int = 0
        self.min_onset_frames: int = 3
        self.hangover_frames: int = 12

    def calibrate_noise(self, noise_audio: np.ndarray):
        """
        Estimate background ambient noise floor from an initial silent/ambient audio chunk.
        :param noise_audio: PCM array of initial background recording (~200-500ms)
        """
        if noise_audio is None or len(noise_audio) < self.frame_size:
            return

        frames = self._slice_frames(noise_audio)
        if len(frames) == 0:
            return

        energies = [self._compute_energy(f) for f in frames]
        sfms = [self._compute_sfm(f) for f in frames]

        self.noise_mean_energy = float(np.mean(energies))
        self.noise_std_energy = float(np.std(energies))
        self.noise_mean_sfm = float(np.mean(sfms))
        self.noise_std_sfm = float(np.std(sfms))

        self.energy_threshold = max(1e-4, self.noise_mean_energy + 3.0 * self.noise_std_energy)
        self.sfm_threshold = min(0.50, max(0.20, self.noise_mean_sfm - 2.0 * self.noise_std_sfm))
        self.calibrated = True

    def _slice_frames(self, audio: np.ndarray) -> List[np.ndarray]:
        """Slice 1D audio array into overlapping frames of length frame_size with step hop_size."""
        audio_float = audio.astype(np.float32) / 32768.0
        n_samples = len(audio_float)
        if n_samples < self.frame_size:
            return []

        num_frames = 1 + int(np.floor((n_samples - self.frame_size) / self.hop_size))
        frames = []
        for i in range(num_frames):
            start = i * self.hop_size
            frames.append(audio_float[start:start + self.frame_size])
        return frames

    def _compute_energy(self, frame: np.ndarray) -> float:
        """Short-Time Energy: E = (1/N) * sum(x[n]^2)"""
        return float(np.mean(frame ** 2))

    def _compute_zcr(self, frame: np.ndarray) -> float:
        """Zero-Crossing Rate: ZCR = 1/(2N) * sum(|sgn(x[n]) - sgn(x[n-1])|)"""
        signs = np.sign(frame)

        signs[signs == 0] = 1
        return float(0.5 * np.mean(np.abs(signs[1:] - signs[:-1])))

    def _compute_sfm(self, frame: np.ndarray) -> float:
        """
        Spectral Flatness Measure (Wiener Entropy):
        SFM = exp( (1/K) * sum(ln(P[k])) ) / ( (1/K) * sum(P[k]) )
        Values close to 1 -> white noise / flat spectrum
        Values close to 0 -> tonal / harmonic speech resonances
        """

        w = np.hamming(len(frame))
        windowed = frame * w
        fft_mag = np.abs(np.fft.rfft(windowed, 512))
        power_spectrum = (fft_mag ** 2) / len(frame) + 1e-12

        geo_mean = np.exp(np.mean(np.log(power_spectrum)))

        arith_mean = np.mean(power_spectrum)

        if arith_mean == 0:
            return 1.0
        return float(np.clip(geo_mean / arith_mean, 0.0, 1.0))

    def is_frame_active(self, frame: np.ndarray) -> bool:
        """
        Pure mathematical frame-level decision rule.
        Checks if energy and spectral characteristics indicate voiced/unvoiced human speech.
        """
        energy = self._compute_energy(frame)
        sfm = self._compute_sfm(frame)
        zcr = self._compute_zcr(frame)

        high_energy = energy > (self.energy_threshold * 2.0)

        voiced_speech = (energy > self.energy_threshold) and (sfm < self.sfm_threshold)

        unvoiced_consonant = (energy > self.energy_threshold * 0.8) and (zcr > 0.18) and (sfm < 0.6)

        return high_energy or voiced_speech or unvoiced_consonant

    def process_frame(self, frame: np.ndarray) -> str:
        """
        Process single frame through onset and hangover finite state machine.
        :return: 'SPEECH' or 'SILENCE'
        """
        active = self.is_frame_active(frame)

        if self.state == "SILENCE":
            if active:
                self.onset_counter += 1
                if self.onset_counter >= self.min_onset_frames:
                    self.state = "SPEECH"
                    self.hangover_counter = 0
            else:
                self.onset_counter = 0
        elif self.state == "SPEECH":
            if active:
                self.hangover_counter = 0
            else:
                self.hangover_counter += 1
                if self.hangover_counter >= self.hangover_frames:
                    self.state = "SILENCE"
                    self.onset_counter = 0
                    self.hangover_counter = 0

        return self.state

    def process_chunk(self, chunk: np.ndarray) -> List[Tuple[int, str]]:
        """
        Process an audio chunk and return frame-level state transitions.
        :param chunk: Audio PCM chunk (e.g. 100ms)
        :return: List of (frame_start_sample, state)
        """
        frames = self._slice_frames(chunk)
        results = []
        for i, frame in enumerate(frames):
            state = self.process_frame(frame)
            results.append((i * self.hop_size, state))
        return results

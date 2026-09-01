"""
Speaker differentiation module.
Distinguishes between conversational participants anonymously (Speaker A, Speaker B, Speaker C)
using fundamental frequency (F0 pitch tracking via autocorrelation) and spectral centroid features.
No permanent biometric voiceprints or face profiles are stored.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

class SpeakerIdentifier:
    """
    Online acoustic speaker clusterer for turn differentiation.
    """

    def __init__(self, sample_rate: int = 16000, max_speakers: int = 6):
        self.sample_rate = sample_rate
        self.max_speakers = max_speakers

        self.speaker_clusters: List[Dict[str, Any]] = []
        self.distance_threshold: float = 35.0

    def _estimate_f0_frame(self, frame: np.ndarray) -> float:
        """
        Estimate pitch (fundamental frequency) using Normalized Autocorrelation (NACS)
        with center clipping and parabolic interpolation.
        """
        if len(frame) == 0:
            return 0.0

        max_val = np.max(np.abs(frame))
        if max_val < 1e-4:
            return 0.0
        cl = 0.60 * max_val
        clipped = np.sign(frame) * np.maximum(0.0, np.abs(frame) - cl)

        N = len(clipped)
        fft_size = 2 ** int(np.ceil(np.log2(2 * N)))
        X = np.fft.fft(clipped, fft_size)
        r = np.fft.ifft(X * np.conj(X)).real[:N]

        if r[0] == 0:
            return 0.0
        norm_r = r / r[0]

        tau_min = int(self.sample_rate / 500)
        tau_max = int(self.sample_rate / 50)
        tau_max = min(tau_max, N - 2)

        if tau_min >= tau_max:
            return 0.0

        peak_lag = tau_min + int(np.argmax(norm_r[tau_min:tau_max + 1]))

        if norm_r[peak_lag] < 0.35:
            return 0.0

        if 0 < peak_lag < N - 1:
            alpha = norm_r[peak_lag - 1]
            beta = norm_r[peak_lag]
            gamma = norm_r[peak_lag + 1]
            denom = 2.0 * (alpha - 2.0 * beta + gamma)
            delta = (alpha - gamma) / denom if denom != 0 else 0.0
            refined_lag = peak_lag + delta
        else:
            refined_lag = float(peak_lag)

        if refined_lag <= 0:
            return 0.0

        f0 = self.sample_rate / refined_lag
        return float(f0) if 50.0 <= f0 <= 500.0 else 0.0

    def _compute_spectral_centroid(self, frame: np.ndarray) -> float:
        """Calculate the spectral centroid (center of mass of the frequency spectrum)."""
        w = np.hamming(len(frame))
        fft_mag = np.abs(np.fft.rfft(frame * w, 512))
        freqs = np.fft.rfftfreq(512, 1.0 / self.sample_rate)

        sum_mag = np.sum(fft_mag)
        if sum_mag == 0:
            return 0.0
        return float(np.sum(freqs * fft_mag) / sum_mag)

    def extract_voice_features(self, audio: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Extract average voiced F0 and Spectral Centroid across an audio turn.
        :return: (mean_f0, mean_centroid) or None if unvoiced
        """
        audio_float = audio.astype(np.float32) / 32768.0 if audio.dtype != np.float32 else audio
        frame_len = 400
        hop_len = 160

        n_samples = len(audio_float)
        if n_samples < frame_len:
            return None

        num_frames = 1 + int(np.floor((n_samples - frame_len) / hop_len))
        f0_list = []
        centroid_list = []

        for i in range(num_frames):
            start = i * hop_len
            frame = audio_float[start:start + frame_len]
            f0 = self._estimate_f0_frame(frame)
            if f0 > 0.0:
                f0_list.append(f0)
                centroid_list.append(self._compute_spectral_centroid(frame))

        if len(f0_list) < 3:
            return None

        return float(np.median(f0_list)), float(np.mean(centroid_list))

    def identify_speaker(self, audio: np.ndarray) -> str:
        """
        Assign an anonymous speaker identifier (Speaker A, Speaker B, etc.)
        to the incoming speech segment.
        """
        features = self.extract_voice_features(audio)
        if features is None:

            return self.speaker_clusters[0]['id'] if self.speaker_clusters else "Speaker A"

        f0, centroid = features

        if len(self.speaker_clusters) == 0:

            cluster = {'id': 'Speaker A', 'f0': f0, 'centroid': centroid, 'count': 1}
            self.speaker_clusters.append(cluster)
            return cluster['id']

        best_dist = float("inf")
        best_cluster = None

        for c in self.speaker_clusters:

            dist = abs(c['f0'] - f0) + 0.01 * abs(c['centroid'] - centroid)
            if dist < best_dist:
                best_dist = dist
                best_cluster = c

        if best_dist <= self.distance_threshold or len(self.speaker_clusters) >= self.max_speakers:

            best_cluster['f0'] = 0.80 * best_cluster['f0'] + 0.20 * f0
            best_cluster['centroid'] = 0.80 * best_cluster['centroid'] + 0.20 * centroid
            best_cluster['count'] += 1
            return best_cluster['id']
        else:

            speaker_idx = len(self.speaker_clusters)
            speaker_label = f"Speaker {chr(65 + speaker_idx)}"
            new_cluster = {'id': speaker_label, 'f0': f0, 'centroid': centroid, 'count': 1}
            self.speaker_clusters.append(new_cluster)
            return speaker_label

    def reset(self):
        """Clear all active speaker clusters at the end of the meeting."""
        self.speaker_clusters.clear()

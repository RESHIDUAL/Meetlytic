"""
Mel-Frequency Cepstral Coefficients (MFCC) feature extraction implemented from scratch.
Uses pure NumPy for all signal processing transformations:
Pre-emphasis -> Framing -> Hamming Windowing -> FFT -> Periodogram -> Mel Filterbank -> Log -> DCT-II.
"""

import numpy as np
from typing import Tuple

class MFCCExtractor:
    """
    Computes 13-dimensional MFCC vectors for speech acoustic analysis.
    Implements standard HTK/Slaney auditory Mel frequency warping.
    """

    def __init__(self, sample_rate: int = 16000,
                 n_mfcc: int = 13,
                 n_mels: int = 26,
                 n_fft: int = 512,
                 win_length: int = 400,
                 hop_length: int = 160,
                 pre_emphasis: float = 0.97):
        """
        :param sample_rate: Audio sampling frequency in Hz (16000)
        :param n_mfcc: Number of cepstral coefficients to retain (13: C0..C12)
        :param n_mels: Number of triangular Mel filterbanks (26)
        :param n_fft: FFT length (512)
        :param win_length: Window length in samples (400 = 25ms)
        :param hop_length: Hop length in samples (160 = 10ms)
        :param pre_emphasis: First-order high-pass pre-emphasis filter coefficient (0.97)
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.pre_emphasis = pre_emphasis

        self.hamming_window = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(self.win_length) / (self.win_length - 1))

        self.mel_filterbank = self._build_mel_filterbank()

        self.dct_matrix = self._build_dct_matrix()

    def _hz_to_mel(self, hz: np.ndarray) -> np.ndarray:
        """Convert frequency in Hz to perceived pitch in Mel scale."""
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: np.ndarray) -> np.ndarray:
        """Convert Mel scale value back to frequency in Hz."""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_filterbank(self) -> np.ndarray:
        """
        Construct bank of 26 triangular bandpass filters spaced linearly on the Mel scale.
        """
        low_mel = self._hz_to_mel(np.array(0.0))
        high_mel = self._hz_to_mel(np.array(self.sample_rate / 2.0))

        mel_points = np.linspace(low_mel, high_mel, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        bin_indices = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        n_bins = self.n_fft // 2 + 1
        filterbank = np.zeros((self.n_mels, n_bins), dtype=np.float32)

        for m in range(1, self.n_mels + 1):
            f_left = bin_indices[m - 1]
            f_center = bin_indices[m]
            f_right = bin_indices[m + 1]

            if f_center > f_left:
                for k in range(f_left, f_center):
                    filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)

            if f_right > f_center:
                for k in range(f_center, f_right):
                    filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)

        return filterbank

    def _build_dct_matrix(self) -> np.ndarray:
        """
        Construct DCT-II basis matrix for decorrelating filterbank log-energies.
        Formula: C[k, m] = cos( (pi * k / M) * (m + 0.5) )
        """
        dct = np.zeros((self.n_mfcc, self.n_mels), dtype=np.float32)
        for k in range(self.n_mfcc):
            for m in range(self.n_mels):
                dct[k, m] = np.cos(np.pi * k * (m + 0.5) / self.n_mels)
        return dct

    def _apply_pre_emphasis(self, signal: np.ndarray) -> np.ndarray:
        """
        High-pass filter: y[n] = x[n] - alpha * x[n-1].
        Cancels glottal roll-off (-6 dB/octave) to equalize energy across frequencies.
        """
        if len(signal) == 0:
            return signal
        return np.append(signal[0], signal[1:] - self.pre_emphasis * signal[:-1])

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC feature matrix from 1D PCM audio array.
        :param audio: 1D numpy array of audio samples
        :return: 2D numpy array of shape (num_frames, n_mfcc)
        """
        if audio is None or len(audio) < self.win_length:
            return np.zeros((0, self.n_mfcc), dtype=np.float32)

        audio_float = audio.astype(np.float32)
        if np.max(np.abs(audio_float)) > 1.0:
            audio_float = audio_float / 32768.0

        emphasized = self._apply_pre_emphasis(audio_float)

        n_samples = len(emphasized)
        num_frames = 1 + int(np.floor((n_samples - self.win_length) / self.hop_length))
        if num_frames <= 0:
            return np.zeros((0, self.n_mfcc), dtype=np.float32)

        indices = (np.tile(np.arange(0, self.win_length), (num_frames, 1)) +
                   np.tile(np.arange(0, num_frames * self.hop_length, self.hop_length), (self.win_length, 1)).T)
        frames = emphasized[indices]

        windowed = frames * self.hamming_window

        fft_complex = np.fft.rfft(windowed, self.n_fft)
        mag_spectrum = np.abs(fft_complex)
        power_spectrum = (1.0 / self.n_fft) * (mag_spectrum ** 2)

        filter_energies = np.dot(power_spectrum, self.mel_filterbank.T)

        filter_energies = np.where(filter_energies <= 0, 1e-12, filter_energies)
        log_energies = np.log(filter_energies)

        mfcc = np.dot(log_energies, self.dct_matrix.T)

        mfcc -= np.mean(mfcc, axis=0, keepdims=True)

        return mfcc.astype(np.float32)

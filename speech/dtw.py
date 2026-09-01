"""
Dynamic Time Warping (DTW) for variable-length acoustic feature alignment.
Implements dynamic programming with Sakoe-Chiba band constraints to achieve O((N+M)R)
time complexity and prevent pathological warping distortions.
"""

import numpy as np
from typing import Tuple, List, Optional

class DynamicTimeWarping:
    """
    Computes optimal non-linear temporal alignment distance between
    two multi-dimensional acoustic feature sequences (e.g. MFCC matrices).
    """

    def __init__(self, metric: str = "cosine", band_radius: int = 15):
        """
        :param metric: 'cosine' or 'euclidean'
        :param band_radius: Sakoe-Chiba corridor width in frames
        """
        self.metric = metric
        self.band_radius = band_radius

    def _compute_local_cost(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distance matrix between frames of X (N, d) and Y (M, d).
        """
        if self.metric == "cosine":
            norm_X = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
            norm_Y = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10
            X_norm = X / norm_X
            Y_norm = Y / norm_Y

            sim = np.dot(X_norm, Y_norm.T)
            cost = 1.0 - sim
            return np.clip(cost, 0.0, 2.0)
        else:

            diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
            return np.sqrt(np.sum(diff ** 2, axis=-1))

    def compute(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Compute normalized DTW distance and optimal warping path.
        :param X: Query feature sequence of shape (N, feature_dim)
        :param Y: Reference/Template feature sequence of shape (M, feature_dim)
        :return: (normalized_distance, warping_path)
        """
        N = len(X)
        M = len(Y)

        if N == 0 or M == 0:
            return float("inf"), []

        cost = self._compute_local_cost(X, Y)

        D = np.full((N + 1, M + 1), np.inf, dtype=np.float32)
        D[0, 0] = 0.0

        slope = M / N if N > 0 else 1.0

        for i in range(1, N + 1):
            center_j = int(round(i * slope))
            j_start = max(1, center_j - self.band_radius)
            j_end = min(M + 1, center_j + self.band_radius + 1)

            for j in range(j_start, j_end):
                d_local = cost[i - 1, j - 1]

                min_prev = min(D[i - 1, j - 1],
                               D[i - 1, j],
                               D[i, j - 1])
                D[i, j] = d_local + min_prev

        total_cost = D[N, M]
        if np.isinf(total_cost):

            return self._compute_unconstrained(cost, N, M)

        normalized_distance = float(total_cost / (N + M))

        path = []
        i, j = N, M
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            if i == 1 and j == 1:
                break
            candidates = [
                (D[i - 1, j - 1], i - 1, j - 1),
                (D[i - 1, j],     i - 1, j),
                (D[i, j - 1],     i,     j - 1)
            ]
            candidates.sort(key=lambda x: x[0])
            _, i, j = candidates[0]

        path.reverse()
        return normalized_distance, path

    def _compute_unconstrained(self, cost: np.ndarray, N: int, M: int) -> Tuple[float, List[Tuple[int, int]]]:
        """Full DP fallback without band constraints."""
        D = np.full((N + 1, M + 1), np.inf, dtype=np.float32)
        D[0, 0] = 0.0
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
        norm_dist = float(D[N, M] / (N + M))
        return norm_dist, []

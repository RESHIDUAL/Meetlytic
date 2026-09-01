"""
Keyword spotter using acoustic template matching via MFCC + Dynamic Time Warping.
Enables limited-vocabulary keyword spotting from scratch without neural networks.
"""

import os
import glob
import numpy as np
from typing import Dict, List, Tuple, Optional
from speech.features import MFCCExtractor
from speech.dtw import DynamicTimeWarping

class KeywordSpotter:
    """
    Template-matching keyword recognition engine.
    Matches audio segments against enrolled spoken templates.
    """

    def __init__(self, templates_dir: str = "speech/templates",
                 threshold: float = 1.15,
                 sample_rate: int = 16000):
        self.templates_dir = templates_dir
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.mfcc_extractor = MFCCExtractor(sample_rate=sample_rate)
        self.dtw = DynamicTimeWarping(metric="cosine", band_radius=15)

        self.templates: Dict[str, List[np.ndarray]] = {}
        self.load_templates()

    def load_templates(self):
        """
        Load all stored .npy MFCC template arrays from disk.
        Expects directory structure: templates_dir/<keyword>/template_*.npy
        """
        self.templates.clear()
        if not os.path.exists(self.templates_dir):
            os.makedirs(self.templates_dir, exist_ok=True)
            return

        for keyword_dir in os.listdir(self.templates_dir):
            full_dir = os.path.join(self.templates_dir, keyword_dir)
            if os.path.isdir(full_dir):
                npy_files = glob.glob(os.path.join(full_dir, "*.npy"))
                loaded_for_keyword = []
                for npy_file in npy_files:
                    try:
                        arr = np.load(npy_file)
                        if arr.ndim == 2 and arr.shape[1] == 13:
                            loaded_for_keyword.append(arr)
                    except Exception as e:
                        print(f"[KeywordSpotter] Error loading template {npy_file}: {e}")

                if loaded_for_keyword:
                    self.templates[keyword_dir.lower()] = loaded_for_keyword

    def add_template_from_audio(self, keyword: str, audio: np.ndarray) -> bool:
        """
        Enroll a new spoken keyword template from raw audio.
        Extracts MFCCs and persists to disk as .npy array.
        """
        keyword_clean = keyword.strip().lower()
        if not keyword_clean or len(audio) < 1600:
            return False

        mfcc = self.mfcc_extractor.extract(audio)
        if len(mfcc) < 5:
            return False

        target_dir = os.path.join(self.templates_dir, keyword_clean)
        os.makedirs(target_dir, exist_ok=True)
        existing = glob.glob(os.path.join(target_dir, "template_*.npy"))
        template_idx = len(existing) + 1
        save_path = os.path.join(target_dir, f"template_{template_idx}.npy")
        np.save(save_path, mfcc)

        if keyword_clean not in self.templates:
            self.templates[keyword_clean] = []
        self.templates[keyword_clean].append(mfcc)
        return True

    def spot_keywords(self, audio_segment: np.ndarray) -> List[Dict[str, float]]:
        """
        Analyze an audio segment and detect enrolled keywords.
        Returns list of {'keyword': str, 'confidence': float, 'distance': float}.
        """
        if len(self.templates) == 0 or len(audio_segment) < 2000:
            return []

        segment_mfcc = self.mfcc_extractor.extract(audio_segment)
        if len(segment_mfcc) < 10:
            return []

        detected = []

        for keyword, template_list in self.templates.items():
            min_dist = float("inf")
            for tmpl in template_list:

                if len(segment_mfcc) > len(tmpl) * 1.6:
                    dist = self._sliding_window_dtw(segment_mfcc, tmpl)
                else:
                    dist, _ = self.dtw.compute(segment_mfcc, tmpl)
                if dist < min_dist:
                    min_dist = dist

            if min_dist <= self.threshold:

                confidence = float(np.clip(1.0 - (min_dist / self.threshold), 0.0, 1.0))
                detected.append({
                    'keyword': keyword,
                    'confidence': confidence,
                    'distance': min_dist
                })

        detected.sort(key=lambda x: x['confidence'], reverse=True)
        return detected

    def _sliding_window_dtw(self, segment_mfcc: np.ndarray, template_mfcc: np.ndarray) -> float:
        """
        Slide template across longer speech segment to spot keyword occurrence.
        """
        t_len = len(template_mfcc)
        step = max(3, t_len // 4)
        min_distance = float("inf")

        for start in range(0, len(segment_mfcc) - t_len + 1, step):
            sub_seq = segment_mfcc[start:start + int(t_len * 1.3)]
            dist, _ = self.dtw.compute(sub_seq, template_mfcc)
            if dist < min_distance:
                min_distance = dist

        return min_distance

    @property
    def enrolled_keywords(self) -> List[str]:
        return sorted(list(self.templates.keys()))

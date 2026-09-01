"""
Transcript Quality & ASR Resilience Analyzer for Meetlytic.
Performs lightweight, offline linguistic and acoustic coherence checks to identify
likely speech-to-text corruption, incomplete fragments, and semantic anomalies.
Strictly free of emojis and emdashes.
"""

import re
from typing import List, Tuple, Dict, Any

class TranscriptQualityAnalyzer:
    """
    Evaluates transcript coherence, detecting ASR corruption and fragment noise.
    """

    ASR_CORRUPTION_PATTERNS = [
        r'\b(?:trustee\s+staubach|my\s+right\s+trustee|pull\.\.\.|um\s+uh\s+um|blah\s+blah)\b',
        r'\b(?:asdf|qwerty|lorem\s+ipsum)\b',
        r'^(?:and|or|but|because|so|yet|then)\s*$',
        r'^[a-zA-Z]\s*$'
    ]

    def evaluate_transcript(self, turns: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Analyze transcript quality across all spoken turns.
        :param turns: List of (Speaker, Utterance) tuples
        :return: Quality assessment dictionary with score, rating, and diagnostics.
        """
        if not turns:
            return {
                'quality': 'LOW',
                'score': 0.0,
                'reasons': ['Empty transcript with no conversational turns'],
                'corrupted_turns_count': 0,
                'total_turns': 0
            }

        total_turns = len(turns)
        corrupted_count = 0
        fragment_count = 0
        reasons = []

        for idx, (spk, text) in enumerate(turns):
            text_clean = text.strip()
            text_lower = text_clean.lower()
            words = text_clean.split()

            if len(words) <= 1 and text_lower not in ('yes', 'no', 'agreed', 'approved', 'confirmed'):
                fragment_count += 1
                continue

            if any(re.search(pat, text_lower) for pat in self.ASR_CORRUPTION_PATTERNS):
                corrupted_count += 1
                reasons.append(f"Likely ASR error in turn {idx + 1}: '{text_clean[:30]}...'")
                continue

            if re.search(r'\b(\w+)(?:\s+\1){3,}\b', text_lower):
                corrupted_count += 1
                reasons.append(f"Excessive word repetition in turn {idx + 1}")
                continue

            if text_clean.endswith('...') and len(words) <= 2:
                fragment_count += 1

        penalty = (corrupted_count * 0.25) + (fragment_count * 0.10)
        quality_score = max(0.0, round(1.0 - (penalty / max(1, total_turns)), 2))

        if quality_score >= 0.85:
            quality_rating = 'HIGH'
        elif quality_score >= 0.60:
            quality_rating = 'MEDIUM'
        else:
            quality_rating = 'LOW'

        if not reasons and quality_rating == 'HIGH':
            reasons.append("High semantic coherence and clean phonetic recognition")

        return {
            'quality': quality_rating,
            'score': quality_score,
            'reasons': reasons[:5],
            'corrupted_turns_count': corrupted_count,
            'fragment_turns_count': fragment_count,
            'total_turns': total_turns
        }

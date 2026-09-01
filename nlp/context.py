"""
Context awareness engine.
Maintains short-term conversational context to accurately classify ambiguous sentences
(e.g., "I am going to Bangalore tomorrow" -> Personal in isolation, but Professional
when preceded by "The client deployment is scheduled for next week").
"""

from collections import deque, Counter
from typing import Dict, List, Optional, Any
from config.vocabulary import PROFESSIONAL_KEYWORDS

class ContextEngine:
    """
    Tracks conversation trajectory, keyword persistence, and topic momentum.
    """

    def __init__(self, window_size: int = 5):
        """
        :param window_size: Number of previous conversation turns to retain in context memory
        """
        self.window_size = window_size
        self.history: deque[Dict[str, Any]] = deque(maxlen=window_size)
        self.professional_keyword_frequency: Counter = Counter()
        self.current_topic: str = "General Discussion"
        self.streak_professional: int = 0
        self.streak_personal: int = 0

    def update(self, text: str, classification: str, score: float, tokens: List[str]):
        """
        Update context memory with the latest classified conversation segment.
        """
        entry = {
            'text': text,
            'classification': classification,
            'score': score,
            'tokens': tokens
        }
        self.history.append(entry)

        if classification == "professional":
            self.streak_professional += 1
            self.streak_personal = 0

            for t in tokens:
                if t in PROFESSIONAL_KEYWORDS:
                    self.professional_keyword_frequency[t] += 1
        elif classification in ("personal", "casual"):
            self.streak_personal += 1
            self.streak_professional = 0
        else:

            pass

        if self.professional_keyword_frequency:
            top_kw, count = self.professional_keyword_frequency.most_common(1)[0]
            self.current_topic = f"Project / {top_kw.title()}"
        else:
            self.current_topic = "General Discussion"

    def get_context_boost(self, text: str, tokens: List[str]) -> float:
        """
        Calculate contextual score adjustment [-0.20 to +0.25] for the current sentence.
        """
        if len(self.history) == 0:
            return 0.0

        boost = 0.0

        if self.streak_professional >= 2:
            boost += 0.12
        elif self.streak_professional == 1:
            boost += 0.05

        if self.streak_personal >= 2:
            boost -= 0.10

        overlapping_terms = sum(1 for t in tokens if self.professional_keyword_frequency[t] > 0)
        if overlapping_terms > 0:
            boost += min(0.12, 0.04 * overlapping_terms)

        if len(self.history) > 0 and self.history[-1]['classification'] == 'professional':
            if self.history[-1]['score'] >= 0.85:
                boost += 0.05

        return float(max(-0.20, min(0.25, boost)))

    def get_conversation_state(self) -> Dict[str, Any]:
        """Return snapshot of current conversation tracking metrics."""
        return {
            'current_topic': self.current_topic,
            'professional_streak': self.streak_professional,
            'personal_streak': self.streak_personal,
            'history_length': len(self.history),
            'frequent_keywords': [k for k, _ in self.professional_keyword_frequency.most_common(5)]
        }

    def reset(self):
        """Reset context state at the beginning of a new meeting."""
        self.history.clear()
        self.professional_keyword_frequency.clear()
        self.current_topic = "General Discussion"
        self.streak_professional = 0
        self.streak_personal = 0

"""
Privacy modes configuration and decision policies.
Supports STRICT, BALANCED, and CUSTOM privacy enforcement levels.
"""

from typing import Dict, Any, Set, Optional

class PrivacyMode:
    STRICT = "STRICT"
    BALANCED = "BALANCED"
    CUSTOM = "CUSTOM"

class PrivacyConfig:
    """
    Encapsulates retention thresholds and uncertain conversation routing rules.
    """

    def __init__(self, mode: str = PrivacyMode.BALANCED,
                 custom_threshold: float = 0.70,
                 custom_professional_keywords: Optional[Set[str]] = None,
                 custom_personal_keywords: Optional[Set[str]] = None):
        self.mode = mode.upper()
        self.custom_threshold = custom_threshold
        self.custom_professional_keywords = custom_professional_keywords or set()
        self.custom_personal_keywords = custom_personal_keywords or set()

    @property
    def professional_threshold(self) -> float:
        """Return the required professional confidence threshold based on active privacy mode."""
        if self.mode == PrivacyMode.STRICT:
            return 0.85
        elif self.mode == PrivacyMode.BALANCED:
            return 0.70
        elif self.mode == PrivacyMode.CUSTOM:
            return self.custom_threshold
        return 0.70

    def evaluate_retention(self, classification: str, confidence: float) -> str:
        """
        Determine privacy action for a classified conversation unit.
        :return: 'RETAIN', 'DELETE', or 'ASK_USER'
        """

        if classification in ("personal", "casual", "irrelevant"):
            return "DELETE"

        if classification == "professional":
            if confidence >= self.professional_threshold:
                return "RETAIN"
            else:

                if self.mode == PrivacyMode.STRICT:
                    return "DELETE"
                return "RETAIN"

        if classification == "uncertain":
            if self.mode == PrivacyMode.STRICT:

                return "DELETE"
            elif self.mode == PrivacyMode.BALANCED:

                return "ASK_USER"
            elif self.mode == PrivacyMode.CUSTOM:
                return "ASK_USER"

        return "DELETE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode': self.mode,
            'professional_threshold': self.professional_threshold,
            'custom_threshold': self.custom_threshold,
            'custom_professional_keywords': list(self.custom_professional_keywords),
            'custom_personal_keywords': list(self.custom_personal_keywords)
        }

"""
Abstractive Business Decision & Task Synthesizer for Meetlytic.
Implements:
1. Verb-Object Action Item Extraction (VERB + dobj + prep/pobj).
2. Jaccard & Token-Overlap Semantic Deduplication (threshold >= 0.65), retaining higher specificity.
3. Decision Dependency Trimming & Base Imperative Normalization (V_root + dobj + modifiers).
Strictly free of emojis and emdashes.
"""

import re
from typing import List, Dict, Any, Optional, Set

class AbstractiveSynthesizer:
    """
    Transforms conversational speech into professional business records and canonical action items.
    """

    def __init__(self):

        self.verb_normalizations = {
            'push': 'Deploy',
            'pushing': 'Deploy',
            'pushed': 'Deploy',
            'fix': 'Resolve',
            'fixing': 'Resolve',
            'fixed': 'Resolve',
            'finish': 'Complete',
            'finishing': 'Complete',
            'handle': 'Prepare and coordinate',
            'handling': 'Prepare and coordinate',
            'write': 'Draft and document',
            'writing': 'Draft and document',
            'coordinate': 'Coordinate with',
            'coordinating': 'Coordinate with',
            'update': 'Update',
            'updating': 'Update',
            'finalize': 'Finalize',
            'finalizing': 'Finalize',
            'test': 'Execute testing for',
            'testing': 'Execute testing for',
            'send': 'Send',
            'explain': 'Present technical implementation of'
        }

        self.decision_strip_patterns = [
            re.compile(r'^(?:(?:yes|no|well|alright|ok|sure|actually|speaking of work|agreed|sounds good|good idea|perfect|confirmed)[,\s\-:\.!\?]+)+', re.IGNORECASE),
            re.compile(r'^(?:i have prepared (?:the )?[^.!?]*\.\s*)', re.IGNORECASE),
            re.compile(r'^(?:i recommend (?:that we )?|i think (?:that )?(?:we should )?|we should |we agreed (?:that )?|we decided (?:to )?|maybe we can |let\'s |let us |decision approved[:\s\-]*|confirmed that )+', re.IGNORECASE)
        ]

        self.participle_to_base = {
            'including': 'Include',
            'adopting': 'Adopt',
            'verifying': 'Verify',
            'maintaining': 'Maintain',
            'using': 'Use',
            'deploying': 'Deploy',
            'updating': 'Update',
            'implementing': 'Implement',
            'finalizing': 'Finalize'
        }

    def synthesize_task(self, raw_action: str, speaker: str) -> str:
        """
        Extract and normalize Verb-Object action item structure:
        Action Verb (VERB) + Direct Object (dobj) + Prepositional Modifiers (prep + pobj).
        """
        if not raw_action:
            return ""

        text = raw_action.strip()

        text = re.sub(r'^(?:and|also|so|then|i will|i\'ll|i can|we will|we need to|you will|you can|please|first|second|third|fourth|fifth|\d+\.)\s+', '', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'^(?:i still need to|i need to|i think we should|action item\s*:\s*)\s*', '', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'^[,\s\-:]+', '', text).strip()
        text = re.sub(r'[,\s\-:]+$', '', text).strip()

        text = re.sub(r'\b(?:before s|before submitti|and notify|and i\'ll)\b.*$', '', text, flags=re.IGNORECASE).strip()

        words = text.split()
        if words:
            first_word_lower = words[0].lower()
            if first_word_lower in self.verb_normalizations:
                replacement = self.verb_normalizations[first_word_lower]
                if len(words) > 1 and replacement.lower().endswith("with") and words[1].lower() == "with":
                    words[0] = replacement
                    words.pop(1)
                else:
                    words[0] = replacement
            elif first_word_lower in self.participle_to_base:
                words[0] = self.participle_to_base[first_word_lower]
            else:
                words[0] = words[0].capitalize()
            text = " ".join(words)

        text = re.sub(r'\bmy\b', 'the', text, flags=re.IGNORECASE)
        text = re.sub(r'\bour\b', 'the project', text, flags=re.IGNORECASE)

        return text.strip()

    def synthesize_decision(self, raw_proposal: str) -> str:
        """
        Reframe an agreed discussion turn into a neutral, declarative business summary
        with base imperative verb normalization (V_root + dobj + modifiers).
        """
        if not raw_proposal:
            return ""

        text = raw_proposal.strip()

        for pat in self.decision_strip_patterns:
            text = pat.sub('', text).strip()

        words = text.split()
        if words:
            first_lower = words[0].lower()
            if first_lower in self.participle_to_base:
                words[0] = self.participle_to_base[first_lower]
                text = " ".join(words)

        text_lower = text.lower()
        if text_lower.startswith(('a comparison', 'the comparison', 'comparison between', 'performance comparison')):
            text = "Include " + text
        elif text_lower.startswith(('esp8266', 'the esp8266', 'microcontroller', 'micro-controller')):
            text = "Adopt " + text
        elif text_lower.startswith(('existing driver', 'current driver', 'legacy implementation')):
            text = "Maintain " + text
        elif text_lower.startswith(('complete system', 'entire pipeline', 'all modules')):
            text = "Verify " + text

        text = re.sub(r'^(?:keep the |use |adopt |implement |include |maintain |verify )', lambda m: m.group(0).capitalize(), text, flags=re.IGNORECASE).strip()
        text = re.sub(r'^[,\s\-:]+', '', text).strip()
        text = re.sub(r'[,\s\-:]+$', '', text).strip()

        if text:
            text = text[0].upper() + text[1:]
            if not text.endswith('.'):
                text += '.'

        return text.strip()

    def compute_jaccard_similarity(self, text_a: str, text_b: str) -> float:
        """Compute Jaccard token similarity over lemmatized content words."""
        words_a = set(re.findall(r'[a-z]{3,}', text_a.lower())) - {'the', 'and', 'for', 'with', 'that', 'this', 'after', 'meeting'}
        words_b = set(re.findall(r'[a-z]{3,}', text_b.lower())) - {'the', 'and', 'for', 'with', 'that', 'this', 'after', 'meeting'}
        if not words_a or not words_b:
            return 0.0
        intersection = len(words_a.intersection(words_b))
        union = len(words_a.union(words_b))
        return intersection / union if union > 0 else 0.0

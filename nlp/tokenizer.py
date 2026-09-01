"""
Self-contained custom tokenizer, rule-based stemmer, and sentence splitter.
Implements lexical normalization without external NLP dependencies (no NLTK, no spaCy).
"""

import re
from typing import List, Set, Optional
from config.vocabulary import STOP_WORDS

class SimpleTokenizer:
    """
    Rule-based tokenizer and morphological stemmer.
    """

    def __init__(self, stop_words: Optional[Set[str]] = None):
        self.stop_words = stop_words if stop_words is not None else STOP_WORDS

        self.stem_suffixes = [
            ('ization', 'ize'),
            ('ational', 'ate'),
            ('fulness', 'ful'),
            ('ousness', 'ous'),
            ('iveness', 'ive'),
            ('tional', 'tion'),
            ('biliti', 'ble'),
            ('lessly', 'less'),
            ('ements', ''),
            ('ement', ''),
            ('ments', ''),
            ('ment', ''),
            ('ating', 'ate'),
            ('izing', 'ize'),
            ('ation', 'ate'),
            ('ities', 'y'),
            ('ness', ''),
            ('able', ''),
            ('ible', ''),
            ('less', ''),
            ('ings', ''),
            ('ing', ''),
            ('ies', 'y'),
            ('ied', 'y'),
            ('tion', ''),
            ('sion', ''),
            ('ful', ''),
            ('ous', ''),
            ('ive', ''),
            ('est', ''),
            ('ers', ''),
            ('er', ''),
            ('ed', ''),
            ('ly', ''),
            ('es', ''),
            ('s', ''),
        ]

    def _clean_text(self, text: str) -> str:
        """Lowercases and strips non-alphanumeric punctuation while retaining hyphens within words."""
        text = text.lower()

        text = re.sub(r'[^a-z0-9\s\-_]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _rule_stem(self, word: str) -> str:
        """
        Lightweight Porter-style rule-based suffix stripper.
        Ensures stem retains a valid root length of >= 3 characters.
        """
        if len(word) <= 3:
            return word

        for suffix, replacement in self.stem_suffixes:
            if word.endswith(suffix):
                root = word[:-len(suffix)]
                if len(root) >= 3:
                    return root + replacement

        return word

    def tokenize(self, text: str, remove_stop_words: bool = True, stem: bool = False) -> List[str]:
        """
        Tokenize input text into normalized word tokens.
        :param text: Raw input string
        :param remove_stop_words: Strip standard English stop words
        :param stem: Apply morphological suffix stripping
        :return: List of string tokens
        """
        if not text:
            return []

        cleaned = self._clean_text(text)
        raw_tokens = cleaned.split()

        tokens = []
        for token in raw_tokens:
            token = token.strip('-').strip('_')
            if not token or len(token) < 2:
                continue

            if remove_stop_words and token in self.stop_words:
                continue

            if stem:
                token = self._rule_stem(token)

            tokens.append(token)

        return tokens

    def split_sentences(self, text: str) -> List[str]:
        """
        Split a block of conversational text into individual sentence units.
        Handles abbreviations and sentence terminators.
        """
        if not text:
            return []

        raw_splits = re.split(r'[\.\!\?\n]+', text)
        sentences = [s.strip() for s in raw_splits if len(s.strip()) > 3]
        return sentences

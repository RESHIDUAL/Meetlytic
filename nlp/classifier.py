"""
Professional vs. Personal conversation classifier.
Features a dual-layer architecture:
1. First-principles rule-based mathematical relevance scoring.
2. Self-trained Multinomial Naive Bayes classifier with Laplace smoothing.
No external ML frameworks or pre-trained models.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from nlp.tokenizer import SimpleTokenizer
from config.vocabulary import (
    PROFESSIONAL_KEYWORDS,
    PERSONAL_KEYWORDS,
    CASUAL_KEYWORDS,
    ACTION_INDICATORS,
    DECISION_INDICATORS,
    DEADLINE_INDICATORS
)

class MultinomialNaiveBayes:
    """
    Pure Python / NumPy Multinomial Naive Bayes with Laplace smoothing.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes: List[str] = []
        self.class_priors: np.ndarray = np.array([])
        self.feature_log_probs: np.ndarray = np.array([])
        self.vocabulary: Dict[str, int] = {}

    def fit(self, tokenized_corpus: List[List[str]], labels: List[str]):
        """
        Train Naive Bayes parameters from scratch on labeled data.
        """
        self.classes = sorted(list(set(labels)))
        n_classes = len(self.classes)
        class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.vocabulary.clear()
        for doc in tokenized_corpus:
            for token in doc:
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)

        vocab_size = len(self.vocabulary)
        if vocab_size == 0:
            return

        class_doc_counts = np.zeros(n_classes, dtype=np.float32)
        word_counts = np.zeros((n_classes, vocab_size), dtype=np.float32)

        for doc, label in zip(tokenized_corpus, labels):
            c_idx = class_to_idx[label]
            class_doc_counts[c_idx] += 1
            for token in doc:
                t_idx = self.vocabulary[token]
                word_counts[c_idx, t_idx] += 1

        total_docs = len(labels)
        self.class_priors = np.log((class_doc_counts + 1e-10) / total_docs)

        total_words_in_class = np.sum(word_counts, axis=1, keepdims=True)
        smoothed_probs = (word_counts + self.alpha) / (total_words_in_class + self.alpha * vocab_size)
        self.feature_log_probs = np.log(smoothed_probs)

    def predict_log_proba(self, tokens: List[str]) -> Tuple[np.ndarray, int]:
        """Calculate unnormalized log posterior probabilities and return number of known tokens."""
        if len(self.vocabulary) == 0:
            return np.zeros(len(self.classes)), 0

        log_posteriors = self.class_priors.copy()
        matched = 0
        for token in tokens:
            if token in self.vocabulary:
                t_idx = self.vocabulary[token]
                log_posteriors += self.feature_log_probs[:, t_idx]
                matched += 1

        return log_posteriors, matched

    def predict_proba(self, tokens: List[str]) -> Tuple[Dict[str, float], int]:
        """Convert log posteriors into normalized probabilities using Log-Sum-Exp."""
        if len(self.classes) == 0:
            return {'uncertain': 1.0}, 0

        log_posts, matched = self.predict_log_proba(tokens)
        if matched == 0:

            return {c: 1.0 / len(self.classes) for c in self.classes}, 0

        max_log = np.max(log_posts)
        exp_vals = np.exp(log_posts - max_log)
        probs = exp_vals / np.sum(exp_vals)

        return {c: float(probs[i]) for i, c in enumerate(self.classes)}, matched

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model parameters to plain dictionary for JSON export."""
        return {
            'alpha': self.alpha,
            'classes': self.classes,
            'class_priors': self.class_priors.tolist(),
            'feature_log_probs': self.feature_log_probs.tolist(),
            'vocabulary': self.vocabulary
        }

    def from_dict(self, data: Dict[str, Any]):
        """Load model parameters from dictionary."""
        self.alpha = data['alpha']
        self.classes = data['classes']
        self.class_priors = np.array(data['class_priors'], dtype=np.float32)
        self.feature_log_probs = np.array(data['feature_log_probs'], dtype=np.float32)
        self.vocabulary = data['vocabulary']

class ConversationClassifier:
    """
    Main Conversation Classification Engine.
    Combines rule-based mathematical scoring with self-trained Naive Bayes.
    """

    def __init__(self):
        self.tokenizer = SimpleTokenizer()
        self.naive_bayes = MultinomialNaiveBayes(alpha=1.0)
        self.is_nb_trained = False

    def _calculate_sub_scores(self, text: str, tokens: List[str]) -> Dict[str, float]:
        """
        Compute constituent semantic relevance scores from keywords and indicators.
        """
        text_lower = text.lower()

        prof_score = 0.0
        for t in tokens:
            if t in PROFESSIONAL_KEYWORDS:
                prof_score += PROFESSIONAL_KEYWORDS[t]
            else:
                stemmed = self.tokenizer._rule_stem(t)
                if stemmed in PROFESSIONAL_KEYWORDS:
                    prof_score += PROFESSIONAL_KEYWORDS[stemmed] * 0.9

        personal_score = sum(PERSONAL_KEYWORDS.get(t, 0.0) for t in tokens)

        casual_score = sum(CASUAL_KEYWORDS.get(t, 0.0) for t in tokens)

        action_count = sum(1 for ind in ACTION_INDICATORS if ind in text_lower)
        action_score = action_count * 0.50

        decision_count = sum(1 for ind in DECISION_INDICATORS if ind in text_lower)
        decision_score = decision_count * 0.50

        deadline_count = sum(1 for ind in DEADLINE_INDICATORS if ind in text_lower)
        deadline_score = deadline_count * 0.50

        return {
            'professional_keywords': prof_score,
            'personal_keywords': personal_score,
            'casual_keywords': casual_score,
            'action_score': action_score,
            'decision_score': decision_score,
            'deadline_score': deadline_score
        }

    def compute_rule_score(self, text: str, tokens: List[str]) -> Tuple[float, Dict[str, float]]:
        """
        Mathematical scoring formula:
        Positive = ProfKeywords + ActionScore + DecisionScore + DeadlineScore
        Negative = PersonalKeywords + CasualKeywords
        Distinguishes Professional (>=0.70), Casual (0.30-0.49), and Personal (<0.30).
        """
        sub = self._calculate_sub_scores(text, tokens)

        positives = (
            sub['professional_keywords'] +
            sub['action_score'] +
            sub['decision_score'] +
            sub['deadline_score']
        )
        personal_neg = sub['personal_keywords']
        casual_neg = sub['casual_keywords']

        if personal_neg > 0 and positives == 0:

            norm_score = float(1.0 / (1.0 + np.exp(1.2 + 0.8 * personal_neg)))
            norm_score = min(0.25, norm_score)
        elif personal_neg > 0 and positives > 0:

            raw = positives - personal_neg * 1.5
            norm_score = float(1.0 / (1.0 + np.exp(- (0.5 * raw))))
        elif casual_neg > 0 and positives == 0:

            norm_score = float(0.38 - min(0.06, 0.02 * casual_neg))
        elif positives > 0 and personal_neg == 0 and casual_neg == 0:

            norm_score = float(1.0 / (1.0 + np.exp(- (0.9 + 0.7 * positives))))
            norm_score = max(0.72, norm_score)
        elif positives > 0 and casual_neg > 0:

            raw = positives - casual_neg * 0.3
            norm_score = float(1.0 / (1.0 + np.exp(- (0.7 + 0.6 * raw))))
        else:

            norm_score = 0.50

        norm_score = float(np.clip(norm_score, 0.0, 1.0))
        return norm_score, sub

    def classify(self, text: str, context_boost: float = 0.0) -> Dict[str, Any]:
        """
        Classify input conversational sentence.
        :param text: Input sentence
        :param context_boost: Contextual probability adjustment from ContextEngine (-0.25 to +0.25)
        :return: Dict containing category, confidence, sub-scores, and decision method
        """
        tokens = self.tokenizer.tokenize(text, remove_stop_words=True)
        rule_score, sub_scores = self.compute_rule_score(text, tokens)

        if self.is_nb_trained and len(tokens) > 0:
            nb_probs, matched_tokens = self.naive_bayes.predict_proba(tokens)
            nb_prof_prob = nb_probs.get('professional', 0.5)

            if matched_tokens >= 2:

                final_prof_score = 0.55 * rule_score + 0.45 * nb_prof_prob
                method = "Ensemble (Rule + Naive Bayes)"
            elif matched_tokens == 1:

                final_prof_score = 0.75 * rule_score + 0.25 * nb_prof_prob
                method = "Ensemble (Rule-Dominant)"
            else:

                final_prof_score = rule_score
                method = "Mathematical Rule-Based"
        else:
            final_prof_score = rule_score
            method = "Mathematical Rule-Based"

        final_prof_score = float(np.clip(final_prof_score + context_boost, 0.0, 1.0))

        token_weights = self.get_token_weightage(text, tokens)

        if final_prof_score >= 0.70:
            category = "professional"
            confidence = final_prof_score
        elif final_prof_score < 0.30:
            category = "personal"
            confidence = 1.0 - final_prof_score
        elif 0.30 <= final_prof_score < 0.50:
            category = "casual"
            confidence = 1.0 - abs(final_prof_score - 0.40) / 0.15
        else:
            category = "uncertain"
            confidence = 1.0 - abs(final_prof_score - 0.60) / 0.20

        return {
            'text': text,
            'classification': category,
            'professional_score': final_prof_score,
            'confidence': float(np.clip(confidence, 0.0, 1.0)),
            'sub_scores': sub_scores,
            'tokens': tokens,
            'token_weights': token_weights,
            'method': method,
            'context_boost_applied': context_boost
        }

    def get_token_weightage(self, text: str, tokens: List[str]) -> List[Dict[str, Any]]:
        """
        Calculate individual token weights, semantic role, and stress level.
        """
        text_lower = text.lower()
        token_weights = []

        for token in tokens:
            t_lower = token.lower()
            stemmed = self.tokenizer._rule_stem(t_lower)

            if t_lower in PROFESSIONAL_KEYWORDS:
                w = PROFESSIONAL_KEYWORDS[t_lower]
                role = "Technical Keyword"
            elif stemmed in PROFESSIONAL_KEYWORDS:
                w = PROFESSIONAL_KEYWORDS[stemmed] * 0.95
                role = "Technical Stem"
            elif any(t_lower in ind for ind in ACTION_INDICATORS):
                w = 0.90
                role = "Action Commitment"
            elif any(t_lower in ind for ind in DEADLINE_INDICATORS):
                w = 0.95
                role = "Deadline Indicator"
            elif any(t_lower in ind for ind in DECISION_INDICATORS):
                w = 0.90
                role = "Decision Cue"
            elif t_lower in PERSONAL_KEYWORDS:
                w = 0.20
                role = "Personal Topic"
            elif t_lower in CASUAL_KEYWORDS:
                w = 0.35
                role = "Casual Small Talk"
            else:
                w = 0.50
                role = "Context Token"

            is_stressed = bool(w >= 0.85 or (len(token) > 4 and w >= 0.75))
            token_weights.append({
                'token': token,
                'weight': round(float(w), 2),
                'role': role,
                'is_stressed': is_stressed,
                'stress_level': 'High Emphasis' if is_stressed else ('Medium' if w >= 0.65 else 'Standard')
            })

        return token_weights

    def train_on_corpus(self, texts: List[str], labels: List[str]):
        """Train Naive Bayes model on a labeled corpus from scratch."""
        tokenized_corpus = [self.tokenizer.tokenize(t, remove_stop_words=True) for t in texts]
        self.naive_bayes.fit(tokenized_corpus, labels)
        self.is_nb_trained = True

    def save_model(self, filepath: str):
        """Export trained Naive Bayes weights to JSON file."""
        if self.is_nb_trained:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.naive_bayes.to_dict(), f, indent=2)

    def load_model(self, filepath: str) -> bool:
        """Load trained Naive Bayes weights from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.naive_bayes.from_dict(data)
            self.is_nb_trained = True
            return True
        except Exception:
            return False

"""
From-scratch Term Frequency - Inverse Document Frequency (TF-IDF) vectorizer.
Supports incremental vocabulary updates, sublinear term scaling, smoothed IDF,
and cosine similarity computation using pure NumPy.
"""

import numpy as np
from typing import Dict, List, Set, Optional

class TFIDFVectorizer:
    """
    Pure Python/NumPy TF-IDF vectorizer.
    No scikit-learn or external machine learning dependencies.
    """

    def __init__(self, use_sublinear_tf: bool = True, smooth_idf: bool = True):
        self.use_sublinear_tf = use_sublinear_tf
        self.smooth_idf = smooth_idf

        self.vocabulary: Dict[str, int] = {}
        self.document_frequencies: Dict[str, int] = {}
        self.n_documents: int = 0
        self._idf_vector: Optional[np.ndarray] = None

    def fit(self, tokenized_docs: List[List[str]]):
        """
        Build vocabulary and compute document frequencies from a list of tokenized documents.
        """
        self.vocabulary.clear()
        self.document_frequencies.clear()
        self.n_documents = len(tokenized_docs)

        for doc in tokenized_docs:
            unique_terms = set(doc)
            for term in unique_terms:
                if term not in self.vocabulary:
                    self.vocabulary[term] = len(self.vocabulary)
                    self.document_frequencies[term] = 1
                else:
                    self.document_frequencies[term] += 1

        self._recompute_idf()

    def _recompute_idf(self):
        """Compute the inverse document frequency vector across the current vocabulary."""
        vocab_size = len(self.vocabulary)
        self._idf_vector = np.zeros(vocab_size, dtype=np.float32)

        for term, idx in self.vocabulary.items():
            df = self.document_frequencies.get(term, 0)
            if self.smooth_idf:

                idf = np.log((1.0 + self.n_documents) / (1.0 + df)) + 1.0
            else:
                idf = np.log(self.n_documents / max(1, df))
            self._idf_vector[idx] = idf

    def update_incremental(self, tokenized_doc: List[str]):
        """
        Incrementally add a single document's terms without re-indexing all historical documents.
        """
        self.n_documents += 1
        unique_terms = set(tokenized_doc)
        for term in unique_terms:
            if term not in self.vocabulary:
                self.vocabulary[term] = len(self.vocabulary)
                self.document_frequencies[term] = 1
            else:
                self.document_frequencies[term] += 1

        self._recompute_idf()

    def transform(self, tokenized_docs: List[List[str]]) -> np.ndarray:
        """
        Convert tokenized documents into L2-normalized TF-IDF feature matrix.
        :return: 2D numpy array of shape (n_docs, vocab_size)
        """
        if self._idf_vector is None or len(self.vocabulary) == 0:
            return np.zeros((len(tokenized_docs), 0), dtype=np.float32)

        vocab_size = len(self.vocabulary)
        matrix = np.zeros((len(tokenized_docs), vocab_size), dtype=np.float32)

        for i, doc in enumerate(tokenized_docs):
            if not doc:
                continue

            counts: Dict[int, int] = {}
            for term in doc:
                if term in self.vocabulary:
                    idx = self.vocabulary[term]
                    counts[idx] = counts.get(idx, 0) + 1

            for idx, count in counts.items():
                if self.use_sublinear_tf:
                    tf = 1.0 + np.log(count)
                else:
                    tf = float(count)
                matrix[i, idx] = tf * self._idf_vector[idx]

            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] /= norm

        return matrix

    def transform_single(self, tokens: List[str]) -> np.ndarray:
        """Transform a single tokenized sentence into a 1D vector."""
        return self.transform([tokens])[0]

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two feature vectors."""
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    @property
    def feature_names(self) -> List[str]:
        """Return vocabulary terms sorted by their index."""
        names = ["" for _ in range(len(self.vocabulary))]
        for term, idx in self.vocabulary.items():
            names[idx] = term
        return names

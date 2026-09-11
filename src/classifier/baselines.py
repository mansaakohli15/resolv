"""
Classifier Baselines for Resolv:
- Baseline 1: Majority Class Classifier (trivial lower bound)
- Baseline 2: Standard TF-IDF + Logistic Regression (standard bag-of-words)
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.config import RANDOM_SEED, INTENT_TAXONOMY


class MajorityClassBaseline:
    """
    Trivial baseline that always predicts the most frequent class in the training set.
    """
    def __init__(self):
        self.majority_class: str = "Delivery Issue"
        self.classes_: np.ndarray = np.array(INTENT_TAXONOMY)

    def fit(self, X: pd.Series, y: pd.Series) -> "MajorityClassBaseline":
        value_counts = y.value_counts()
        self.majority_class = value_counts.index[0]
        self.classes_ = np.array(sorted(y.unique()))
        return self

    def predict(self, X: pd.Series) -> np.ndarray:
        return np.array([self.majority_class] * len(X))

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        probas = np.zeros((len(X), len(self.classes_)))
        maj_idx = np.where(self.classes_ == self.majority_class)[0][0]
        probas[:, maj_idx] = 1.0
        return probas


class SimpleTfidfLogisticBaseline:
    """
    Simple standard TF-IDF unigram + Logistic Regression baseline without class weighting.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 1),
            max_features=5000,
        )
        self.model = LogisticRegression(
            C=1.0,
            max_iter=500,
            random_state=RANDOM_SEED,
            class_weight=None,
        )
        self.classes_ = None

    def fit(self, X: pd.Series, y: pd.Series) -> "SimpleTfidfLogisticBaseline":
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: pd.Series) -> np.ndarray:
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        X_vec = self.vectorizer.transform(X)
        return self.model.predict_proba(X_vec)

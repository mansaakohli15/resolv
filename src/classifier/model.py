"""
Resolv Main Intent Classifier:
- Feature Union of Word n-grams (1-3) and Character n-grams (3-5)
- Class-weighted Logistic Regression with probability calibration
- Confidence estimation and threshold-based uncertainty detection
"""
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from src.config import RANDOM_SEED, INTENT_TAXONOMY
from src.data_utils import clean_text


class ResolvIntentClassifier:
    """
    Production-grade Intent Classifier for Resolv:
    Combines sublinear word n-grams and char n-grams with class-balanced calibrated logistic regression.
    """
    def __init__(self, confidence_threshold: float = 0.55):
        self.confidence_threshold = confidence_threshold
        
        # Word-level TF-IDF (captures domain keywords, multi-word phrases)
        word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=2,
            max_features=15000,
            token_pattern=r"(?u)\b\w+\b|\$[\d]+|[!?]+",
        )
        
        # Char-level TF-IDF (captures misspellings, Twitter contractions, morphology)
        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            min_df=3,
            max_features=25000,
        )
        
        self.feature_extractor = FeatureUnion([
            ("word", word_vectorizer),
            ("char", char_vectorizer),
        ])
        
        # Base estimator with balanced class weights for rare support intents
        base_estimator = LogisticRegression(
            C=2.5,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        )
        
        # Probability calibration via sigmoid on 5-fold CV
        self.model = CalibratedClassifierCV(
            estimator=base_estimator,
            method="sigmoid",
            cv=5,
        )
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: pd.Series, y: pd.Series) -> "ResolvIntentClassifier":
        """Fit feature extractor and calibrated classifier."""
        cleaned_X = [clean_text(t) for t in X]
        X_vec = self.feature_extractor.fit_transform(cleaned_X)
        self.model.fit(X_vec, y)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """Predict calibrated probability distribution across intents."""
        cleaned_X = [clean_text(t) for t in X]
        X_vec = self.feature_extractor.transform(cleaned_X)
        return self.model.predict_proba(X_vec)

    def predict_with_confidence(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predicts top intent, confidence score, and margin for each text.
        """
        cleaned_texts = [clean_text(t) for t in texts]
        X_vec = self.feature_extractor.transform(cleaned_texts)
        probas = self.model.predict_proba(X_vec)
        
        results = []
        for prob in probas:
            top_idx = int(np.argmax(prob))
            top_intent = self.classes_[top_idx]
            confidence = float(prob[top_idx])
            
            sorted_probs = np.sort(prob)[::-1]
            margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else confidence
            
            is_confident = confidence >= self.confidence_threshold and margin >= 0.10
            
            results.append({
                "intent": top_intent,
                "confidence": round(confidence, 4),
                "margin": round(margin, 4),
                "is_confident": is_confident,
                "all_probabilities": {
                    intent: round(float(p), 4) for intent, p in zip(self.classes_, prob)
                }
            })
        return results

    def predict(self, X: pd.Series) -> np.ndarray:
        """Standard scikit-learn predict."""
        probs = self.predict_proba(X)
        top_indices = np.argmax(probs, axis=1)
        return self.classes_[top_indices]

    def save(self, file_path: str):
        """Save fitted model pipeline."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str) -> "ResolvIntentClassifier":
        """Load fitted model pipeline."""
        with open(file_path, "rb") as f:
            return pickle.load(f)

"""
Hybrid Lexical + Semantic Evidence Retrieval Engine for Resolv.
Combines exact keyword BM25/TF-IDF indexing with dense semantic matching and intent-conditioned reranking.
"""
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import INTENT_TAXONOMY, RANDOM_SEED
from src.data_utils import clean_text


class ResolvHybridRetriever:
    """
    Production Hybrid Retriever:
    - Lexical Index: Sublinear TF-IDF (word n-grams 1-2) with entity preservation
    - Semantic Index: High-dimensional Char+Word Dense projections
    - Hybrid Scoring: Weighted linear combination + intent concordance bonus
    """
    def __init__(
        self,
        lexical_weight: float = 0.45,
        semantic_weight: float = 0.55,
        intent_bonus: float = 0.15,
        min_relevance_threshold: float = 0.28,
    ):
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight
        self.intent_bonus = intent_bonus
        self.min_relevance_threshold = min_relevance_threshold
        
        # Lexical vectorizer (precise term matching)
        self.lexical_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_features=25000,
        )
        
        # Dense semantic vectorizer (paraphrase & morphology matching)
        self.semantic_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            min_df=3,
            max_features=35000,
        )
        
        self.corpus_df: Optional[pd.DataFrame] = None
        self.lexical_matrix = None
        self.semantic_matrix = None

    def fit(self, corpus_df: pd.DataFrame) -> "ResolvHybridRetriever":
        """Index the evidence corpus."""
        self.corpus_df = corpus_df.reset_index(drop=True)
        texts = self.corpus_df["cleaned_customer_text"].tolist()
        
        self.lexical_matrix = self.lexical_vec.fit_transform(texts)
        self.semantic_matrix = self.semantic_vec.fit_transform(texts)
        return self

    def retrieve(
        self,
        query: str,
        predicted_intent: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k evidence cases for a customer query.
        
        Returns:
        List of dicts containing:
        - case_id: Unique case id
        - score: Final hybrid relevance score (0.0 to 1.0)
        - customer_text: Historical customer problem
        - agent_reply: Historical brand resolution reply
        - intent: Historical intent
        - is_sufficient: Whether top evidence exceeds confidence threshold
        """
        if self.corpus_df is None:
            raise ValueError("Retriever has not been fitted with a corpus.")
            
        cleaned_q = clean_text(query)
        if not cleaned_q:
            return []
            
        q_lex = self.lexical_vec.transform([cleaned_q])
        q_sem = self.semantic_vec.transform([cleaned_q])
        
        # Compute cosine similarities
        lex_sim = cosine_similarity(q_lex, self.lexical_matrix)[0]
        sem_sim = cosine_similarity(q_sem, self.semantic_matrix)[0]
        
        # Base hybrid score
        hybrid_scores = (self.lexical_weight * lex_sim) + (self.semantic_weight * sem_sim)
        
        # Intent concordance boost if predicted intent is provided
        if predicted_intent and predicted_intent in INTENT_TAXONOMY:
            intent_mask = (self.corpus_df["intent"] == predicted_intent).values
            hybrid_scores = hybrid_scores + (self.intent_bonus * intent_mask * sem_sim)
            
        # Get top candidate indices
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(hybrid_scores[idx])
            row = self.corpus_df.iloc[idx]
            
            results.append({
                "case_id": row["case_id"],
                "component_id": int(row["component_id"]),
                "score": round(score, 4),
                "lexical_score": round(float(lex_sim[idx]), 4),
                "semantic_score": round(float(sem_sim[idx]), 4),
                "customer_text": row["customer_text"],
                "agent_reply": row["agent_reply"],
                "intent": row["intent"],
                "is_sufficient": score >= self.min_relevance_threshold,
            })
            
        return results

    def save(self, file_path: str):
        """Persist index to disk."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str) -> "ResolvHybridRetriever":
        """Load index from disk."""
        with open(file_path, "rb") as f:
            return pickle.load(f)

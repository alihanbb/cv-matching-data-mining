"""Skill similarity matching for improved skill extraction and matching.

Phase 1 Upgrade: Added embedding-based skill similarity for better matching.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SkillSimilarityMatcher:
    """Match skills using embedding-based similarity.
    
    Phase 1 Upgrade: Added for handling unseen skills and fuzzy matching.
    """
    
    def __init__(self, skill_descriptions: dict[str, str], embeddings: np.ndarray):
        """Initialize with skill descriptions and precomputed embeddings.
        
        Args:
            skill_descriptions: Dict mapping skill_id to description
            embeddings: Matrix of shape (n_skills, embedding_dim)
        """
        self.skill_descriptions = skill_descriptions
        self.embeddings = embeddings
        self.skill_ids = list(skill_descriptions.keys())
        
        # Create id to index mapping
        self._id_to_idx = {sid: i for i, sid in enumerate(self.skill_ids)}
    
    def find_similar_skills(
        self,
        skill_id: str,
        top_k: int = 5,
        min_score: float = 0.5,
    ) -> list[tuple[str, float]]:
        """Find similar skills to a given skill.
        
        Args:
            skill_id: Source skill ID
            top_k: Number of similar skills to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of (skill_id, similarity_score) tuples
        """
        if skill_id not in self._id_to_idx:
            return []
        
        idx = self._id_to_idx[skill_id]
        source_embedding = self.embeddings[idx].reshape(1, -1)
        
        similarities = cosine_similarity(source_embedding, self.embeddings)[0]
        
        # Get top-k similar (excluding self)
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for i in top_indices:
            if self.skill_ids[i] == skill_id:
                continue
            score = float(similarities[i])
            if score >= min_score and len(results) < top_k:
                results.append((self.skill_ids[i], score))
        
        return results
    
    def match_text_to_skills(
        self,
        text_embedding: np.ndarray,
        top_k: int = 10,
        min_score: float = 0.3,
    ) -> list[tuple[str, float]]:
        """Match a text embedding to known skills.
        
        Args:
            text_embedding: Embedding of text to match
            top_k: Number of top skills to return
            min_score: Minimum similarity score
            
        Returns:
            List of (skill_id, similarity_score) tuples
        """
        text_embedding = text_embedding.reshape(1, -1)
        similarities = cosine_similarity(text_embedding, self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i in top_indices:
            score = float(similarities[i])
            if score >= min_score:
                results.append((self.skill_ids[i], score))
        
        return results


def create_skill_embeddings_from_lexicon(lex, model=None) -> np.ndarray | None:
    """Create embeddings for all skills in lexicon.
    
    Phase 1 Upgrade: Generate skill embeddings for similarity matching.
    
    Args:
        lex: SkillsLexicon instance
        model: Sentence transformer model (optional)
        
    Returns:
        Embeddings matrix or None if model not available
    """
    if model is None:
        return None
    
    # Create descriptions for each skill
    descriptions = {}
    for skill_id, entry in lex.skills.items():
        # Combine display name and category for richer description
        desc = f"{entry.display} {entry.category} skill"
        descriptions[skill_id] = desc
    
    # Get embeddings
    texts = list(descriptions.values())
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    
    return embeddings


# Phase 1 Upgrade: Skill coverage scoring improvements
def compute_skill_match_score(
    cv_skills: set[str],
    job_skills: set[str],
    skill_similarity: SkillSimilarityMatcher | None = None,
) -> dict[str, float]:
    """Compute detailed skill match scores.
    
    Phase 1 Upgrade: Added similarity-based matching for better coverage.
    
    Args:
        cv_skills: Set of CV skill IDs
        job_skills: Set of job required skill IDs
        skill_similarity: Optional similarity matcher for fuzzy matching
        
    Returns:
        Dict with match metrics
    """
    # Exact matches
    exact_matches = cv_skills & job_skills
    
    # Calculate coverage
    if not job_skills:
        return {
            "exact_coverage": 1.0,
            "fuzzy_coverage": 1.0,
            "total_coverage": 1.0,
            "exact_matches": len(exact_matches),
            "fuzzy_matches": 0,
        }
    
    exact_coverage = len(exact_matches) / len(job_skills)
    
    # Fuzzy matching if similarity available
    fuzzy_matches = set()
    if skill_similarity is not None:
        for job_skill in job_skills:
            if job_skill in exact_matches:
                continue
            similar = skill_similarity.find_similar_skills(job_skill, top_k=3, min_score=0.7)
            for similar_skill, score in similar:
                if similar_skill in cv_skills:
                    fuzzy_matches.add(similar_skill)
    
    fuzzy_coverage = len(fuzzy_matches) / len(job_skills)
    total_coverage = (len(exact_matches) + len(fuzzy_matches)) / len(job_skills)
    
    return {
        "exact_coverage": exact_coverage,
        "fuzzy_coverage": fuzzy_coverage,
        "total_coverage": total_coverage,
        "exact_matches": len(exact_matches),
        "fuzzy_matches": len(fuzzy_matches),
    }
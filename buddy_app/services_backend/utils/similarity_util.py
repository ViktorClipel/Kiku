import numpy as np
from numpy.linalg import norm

def calculate_jaccard_similarity(tags_a: set, tags_b: set) -> float:
    if not tags_a or not tags_b:
        return 0.0
    
    intersection = len(tags_a.intersection(tags_b))
    union = len(tags_a.union(tags_b))
    
    return intersection / union if union > 0 else 0.0

def calculate_cosine_similarity(vec_a, vec_b) -> float:
    if vec_a is None or vec_b is None:
        return 0.0
    
    vec_a = np.asarray(vec_a).flatten()
    vec_b = np.asarray(vec_b).flatten()

    cosine_sim = np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))
    return cosine_sim
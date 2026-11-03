"""Sistema de recomendação de filmes híbrido (SVD + conteúdo) no MovieLens 1M."""

from .config import FINAL_CONFIG, HybridConfig
from .data import Index, Split, load_movielens, temporal_split
from .recommender import HybridRecommender

__all__ = [
    "FINAL_CONFIG",
    "HybridConfig",
    "HybridRecommender",
    "Index",
    "Split",
    "load_movielens",
    "temporal_split",
]
__version__ = "0.4.0"

"""Métricas de ranking top-N: Precision@K, Recall@K, NDCG@K e cobertura."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

from .config import K, RELEVANCE_THRESHOLD
from .data import Index


def relevant_items(df: pd.DataFrame, index: Index, threshold: int = RELEVANCE_THRESHOLD) -> pd.Series:
    """Para cada usuário (linha da matriz), as colunas dos filmes com nota >= `threshold`.

    Usuários sem nenhum filme relevante não aparecem. Notebook: `relevantes_de`.
    """
    relevant = df[df["Rating"] >= threshold]
    return (
        relevant.assign(
            u=index.user_pos[relevant["UserID"]].to_numpy(),
            i=index.item_pos[relevant["MovieID"]].to_numpy(),
        )
        .groupby("u")["i"]
        .apply(np.array)
    )


def seen_mask(R: spmatrix) -> np.ndarray:
    """Máscara densa (usuários x filmes) do que já foi avaliado. Notebook: `vistos_de`."""
    return R.toarray() > 0


def top_k(scores: np.ndarray, seen: np.ndarray, k: int = K) -> np.ndarray:
    """Colunas dos `k` filmes de maior score por linha, ignorando os já vistos.

    Notebook: `top_k`. É a mesma rotina usada na avaliação e no recomendador final,
    o que garante que o modelo em produção é exatamente o que foi avaliado.
    """
    s = np.array(scores, dtype=np.float32, copy=True)
    s[seen] = -np.inf
    s[np.isnan(s)] = -np.inf
    top = np.argpartition(-s, k, axis=1)[:, :k]
    rows = np.arange(s.shape[0])[:, None]
    return top[rows, np.argsort(-s[rows, top], axis=1)]


def evaluate(scores: np.ndarray, name: str, seen: np.ndarray, relevant: pd.Series, k: int = K) -> dict:
    """Precision@k, Recall@k, NDCG@k e cobertura de catálogo. Notebook: `avaliar`.

    - Precision@k: fração da lista que o usuário gostou.
    - Recall@k: fração dos filmes de que ele gostou que entraram na lista.
    - NDCG@k: como a precision, mas acertos no topo valem mais.
    - Cobertura: fração do catálogo que aparece em ao menos uma lista.
    """
    discount = 1 / np.log2(np.arange(2, k + 2))
    top = top_k(scores, seen, k)
    precisions, recalls, ndcgs = [], [], []
    for u, rel in relevant.items():
        hits = np.isin(top[u], rel)
        precisions.append(hits.sum() / k)
        recalls.append(hits.sum() / len(rel))
        ndcgs.append((hits * discount).sum() / discount[: min(len(rel), k)].sum())
    return {
        "Modelo": name,
        f"Precision@{k}": np.mean(precisions),
        f"Recall@{k}": np.mean(recalls),
        f"NDCG@{k}": np.mean(ndcgs),
        "Cobertura": len(np.unique(top[relevant.index.to_numpy()])) / scores.shape[1],
    }

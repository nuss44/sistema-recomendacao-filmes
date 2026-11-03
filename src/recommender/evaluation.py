"""Métricas de ranking top-N: Precision@K, Recall@K, NDCG@K e cobertura."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

from .config import K, RANDOM_STATE, RELEVANCE_THRESHOLD
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


def tie_break_keys(row_id: int, n_items: int, seed: int = RANDOM_STATE) -> np.ndarray:
    """Chaves de desempate do usuário `row_id`: um sorteio próprio de cada usuário,
    sempre o mesmo para a mesma semente.

    - Neutro: nenhum filme vence empates de forma sistemática. Uma ordem única para
      todos os usuários faria o mesmo filme "sortudo" ganhar todos os empates e
      derrubaria a cobertura do catálogo.
    - Reprodutível: não depende do NumPy, do processador nem das outras linhas.
    """
    return np.random.default_rng([seed, row_id]).random(n_items)


def top_k(scores: np.ndarray, seen: np.ndarray, k: int = K, row_ids: np.ndarray | None = None) -> np.ndarray:
    """Colunas dos `k` filmes de maior score por linha, ignorando os já vistos.

    Empates são desfeitos pelo sorteio de cada usuário (`tie_break_keys`). `row_ids`
    identifica o usuário de cada linha (padrão: 0, 1, 2, ...), para que a avaliação de
    todos os usuários e o recomendador, que calcula um usuário por vez, desempatem igual.

    Notebook: `top_k`, com uma diferença deliberada. Lá, `argpartition` + `argsort`
    desempatam de forma arbitrária, e essa ordem muda com a versão do NumPy e até com
    as instruções do processador (AVX2/AVX-512). Nos modelos de conteúdo, em que
    filmes com os mesmos gêneros recebem exatamente o mesmo score, isso mudava as
    métricas de um computador para outro. Onde não há empate, o resultado é idêntico.

    É a mesma rotina usada na avaliação e no recomendador final, o que garante que o
    modelo em produção é exatamente o que foi avaliado.
    """
    s = np.array(scores, dtype=np.float32, copy=True)
    s[seen] = -np.inf
    s[np.isnan(s)] = -np.inf
    if row_ids is None:
        row_ids = np.arange(s.shape[0])
    # k-ésimo maior VALOR de cada linha: é um número, não depende de como os empates caem.
    kth = -np.partition(-s, k - 1, axis=1)[:, k - 1 : k]
    candidates = s >= kth  # inclui todos os empatados na fronteira do top-k
    top = np.empty((s.shape[0], k), dtype=np.intp)
    for row in range(s.shape[0]):
        columns = np.flatnonzero(candidates[row])
        values = s[row, columns]
        if len(columns) == k or len(np.unique(values)) == len(values):
            order = np.argsort(-values, kind="stable")  # sem empate a resolver
        else:
            keys = tie_break_keys(int(row_ids[row]), s.shape[1])[columns]
            order = np.lexsort((keys, -values))  # maior score; empate -> sorteio do usuário
        top[row] = columns[order[:k]]
    return top


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

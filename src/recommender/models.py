"""Modelos de score: cada função recebe os dados de ajuste e devolve uma matriz
(usuários x filmes) em que maior = mais recomendado.

As mesmas funções servem à avaliação e ao recomendador final (notebook, seção
"Modelos"), para que o que vai para produção seja exatamente o que foi avaliado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, spmatrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import RANDOM_STATE, HybridConfig
from .data import Index
from .evaluation import seen_mask


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------
def random_scores(R: spmatrix, seed: int = RANDOM_STATE) -> np.ndarray:
    """Scores aleatórios: limite inferior de qualquer modelo útil."""
    return np.random.default_rng(seed).random(R.shape, dtype=np.float32)


def popularity_scores(R: spmatrix) -> np.ndarray:
    """Quantos usuários avaliaram cada filme (igual para todos os usuários)."""
    popularity = np.asarray((R > 0).sum(axis=0)).ravel().astype(np.float32)
    return np.tile(popularity, (R.shape[0], 1))


# ----------------------------------------------------------------------
# Filtragem colaborativa
# ----------------------------------------------------------------------
def fit_svd(R: spmatrix, n_components: int, random_state: int = RANDOM_STATE) -> tuple[TruncatedSVD, np.ndarray]:
    """Ajusta o TruncatedSVD na matriz de notas (0 = não avaliado).

    Devolve o modelo e os fatores dos usuários (U·Σ).
    """
    model = TruncatedSVD(n_components=n_components, random_state=random_state)
    user_factors = model.fit_transform(R)
    return model, user_factors


def svd_scores(R: spmatrix, n_components: int) -> np.ndarray:
    """Reconstrução da matriz pelo SVD truncado. Notebook: `score_svd_sklearn`."""
    model, user_factors = fit_svd(R, n_components)
    return (user_factors @ model.components_).astype(np.float32)


def surprise_scores(df: pd.DataFrame, params: dict, index: Index) -> np.ndarray:
    """Nota prevista pelo SVD do Surprise para todos os pares usuário x filme.

    Notebook: `score_surprise`. Usado só na comparação de modelos; requer o
    pacote opcional `scikit-surprise` (pip install ".[experiments]").
    """
    try:
        from surprise import SVD, Dataset, Reader
    except ImportError as error:
        raise ImportError(
            "scikit-surprise não está instalado. Ele só é necessário para reproduzir "
            'a comparação com o Surprise: pip install ".[experiments]"'
        ) from error

    reader = Reader(rating_scale=(1, 5))
    trainset = Dataset.load_from_df(df[["UserID", "MovieID", "Rating"]], reader).build_full_trainset()
    model = SVD(**params, random_state=RANDOM_STATE)
    model.fit(trainset)
    # Matriz montada direto dos fatores: média + viés usuário + viés filme + p_u · q_i
    n_factors = params["n_factors"]
    P = np.zeros((index.n_users, n_factors), np.float32)
    bu = np.zeros(index.n_users, np.float32)
    Q = np.zeros((index.n_items, n_factors), np.float32)
    bi = np.zeros(index.n_items, np.float32)
    for raw_uid in df["UserID"].unique():
        inner = trainset.to_inner_uid(raw_uid)
        P[index.user_pos[raw_uid]], bu[index.user_pos[raw_uid]] = model.pu[inner], model.bu[inner]
    for inner in trainset.all_items():
        j = index.item_pos[trainset.to_raw_iid(inner)]
        Q[j], bi[j] = model.qi[inner], model.bi[inner]
    return trainset.global_mean + bu[:, None] + bi[None, :] + P @ Q.T


# ----------------------------------------------------------------------
# Filtragem baseada em conteúdo
# ----------------------------------------------------------------------
def genre_features(movies: pd.DataFrame) -> np.ndarray:
    """Vetores TF-IDF dos gêneros (filmes x gêneros), com norma L2 = 1 por filme.

    Como as linhas já são normalizadas, a similaridade de cosseno entre filmes é
    simplesmente `X @ X.T` — não é preciso guardar a matriz filmes x filmes.
    """
    vectorizer = TfidfVectorizer(token_pattern=r"[^|]+")
    return vectorizer.fit_transform(movies["Genres"].fillna("")).toarray()


def content_weights(R: spmatrix, weighted: bool) -> csr_matrix:
    """Peso de cada filme visto no perfil do usuário.

    - weighted=False: todo filme visto pesa 1.
    - weighted=True: pesa (nota − média do usuário); filmes de que ele não gostou
      passam a afastar recomendações parecidas.
    """
    if not weighted:
        return (R > 0).astype(np.float32)
    coo = R.tocoo()
    mean = np.asarray(R.sum(axis=1)).ravel() / np.maximum(1, R.getnnz(axis=1))
    return csr_matrix((coo.data - mean[coo.row], (coo.row, coo.col)), shape=R.shape)


def content_scores(R: spmatrix, item_features: np.ndarray, weighted: bool) -> np.ndarray:
    """Perfil do usuário = soma ponderada das similaridades com os filmes vistos.

    Notebook: `score_conteudo`. Lá o cálculo é `pesos @ cosine_sim`; aqui é
    `(pesos @ X) @ X.T`, que dá o mesmo resultado sem montar a matriz de
    similaridade 3.883 x 3.883 (~120 MB).
    """
    weights = content_weights(R, weighted)
    profile = np.asarray(weights @ item_features)  # usuários x gêneros
    return np.asarray(profile @ item_features.T, dtype=np.float32)


# ----------------------------------------------------------------------
# Híbrido
# ----------------------------------------------------------------------
def minmax_per_user(scores: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Normaliza os scores de cada usuário para [0, 1], só entre os filmes não vistos.

    Sem isso, o score de conteúdo (soma de similaridades) domina o colaborativo e α perde o efeito.
    """
    s = np.array(scores, dtype=np.float32, copy=True)
    s[seen] = np.nan
    low = np.nanmin(s, axis=1, keepdims=True)
    high = np.nanmax(s, axis=1, keepdims=True)
    return (s - low) / (high - low + 1e-9)


def adaptive_alpha(R: spmatrix, c: float) -> np.ndarray:
    """α_u = n_u / (n_u + c): mais avaliações -> mais peso no colaborativo."""
    n = R.getnnz(axis=1)
    return (n / (n + c)).astype(np.float32)[:, None]


def combine(
    collaborative: np.ndarray,
    content: np.ndarray,
    R: spmatrix,
    seen: np.ndarray,
    alpha: float | None = None,
    c: float | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """α · colaborativo + (1 − α) · conteúdo. Use `alpha` (fixo) OU `c` (adaptativo)."""
    if normalize:
        collaborative = minmax_per_user(collaborative, seen)
        content = minmax_per_user(content, seen)
    a = adaptive_alpha(R, c) if c is not None else alpha
    return a * collaborative + (1 - a) * content


def hybrid_scores(R: spmatrix, item_features: np.ndarray, config: HybridConfig) -> np.ndarray:
    """Híbrido completo a partir de uma configuração. Notebook: `score_hibrido`."""
    seen = seen_mask(R)
    collaborative = svd_scores(R, config.n_components)
    content = content_scores(R, item_features, weighted=config.content_weighted)
    return combine(collaborative, content, R, seen, alpha=config.alpha, c=config.c)

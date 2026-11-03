"""Recomendador final: o híbrido escolhido na validação, pronto para servir.

Diferente da avaliação, que calcula a matriz de scores de todos os usuários de uma
vez, aqui cada pedido calcula só a linha do usuário em questão. O modelo salvo
guarda apenas os fatores do SVD, os vetores de gêneros e as notas conhecidas.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .config import FINAL_CONFIG, HybridConfig
from .data import Index
from .evaluation import top_k
from .models import combine, content_weights, fit_svd, genre_features


class HybridRecommender:
    """SVD truncado + conteúdo (gêneros TF-IDF), combinados com scores normalizados.

    Exemplo:
        rec = HybridRecommender.fit(ratings, movies)
        rec.recommend_for_user(1, n=10)                       # usuário do dataset
        rec.recommend_for_ratings({1: 5, 2355: 4, 1196: 2})   # usuário novo
    """

    def __init__(
        self,
        config: HybridConfig,
        index: Index,
        movies: pd.DataFrame,
        R: csr_matrix,
        svd,
        user_factors: np.ndarray,
        item_features: np.ndarray,
    ):
        self.config = config
        self.index = index
        self.movies = movies.reset_index(drop=True)
        self.R = R
        self.svd = svd
        self.user_factors = user_factors
        self.item_features = item_features
        self.popularity = np.asarray((R > 0).sum(axis=0)).ravel()

    # ------------------------------------------------------------------
    # Treino e persistência
    # ------------------------------------------------------------------
    @classmethod
    def fit(cls, ratings: pd.DataFrame, movies: pd.DataFrame, config: HybridConfig = FINAL_CONFIG) -> "HybridRecommender":
        """Ajusta o recomendador nas avaliações de `ratings`. Notebook: `criar_recomendador`."""
        movies = movies[["MovieID", "Title", "Genres"]].copy()
        movies["Genres"] = movies["Genres"].fillna("")
        index = Index.from_data(ratings, movies)
        R = index.matrix(ratings)
        svd, user_factors = fit_svd(R, config.n_components)
        return cls(config, index, movies, R, svd, user_factors, genre_features(movies))

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path

    @staticmethod
    def load(path: Path | str) -> "HybridRecommender":
        model = joblib.load(path)
        if not isinstance(model, HybridRecommender):
            raise TypeError(f"{path} não contém um HybridRecommender.")
        return model

    # ------------------------------------------------------------------
    # Recomendação
    # ------------------------------------------------------------------
    def recommend_for_user(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """Top-n para um usuário do dataset, excluindo o que ele já avaliou."""
        if user_id not in self.index.user_pos.index:
            raise KeyError(f"UserID {user_id} não encontrado.")
        u = self.index.user_pos[user_id]
        R_row = self.R[u : u + 1]
        collaborative = (self.user_factors[u : u + 1] @ self.svd.components_).astype(np.float32)
        return self._recommend(R_row, collaborative, n, row_id=int(u))

    def recommend_for_ratings(self, ratings: dict[int, float], n: int = 10) -> pd.DataFrame:
        """Top-n para um usuário novo, a partir das notas que ele deu (MovieID -> nota 1–5).

        O vetor do usuário é projetado no espaço do SVD já treinado (fold-in), sem
        re-treinar o modelo. É a mesma conta feita para os usuários do treino.
        """
        R_row = self._ratings_row(ratings)
        collaborative = (self.svd.transform(R_row) @ self.svd.components_).astype(np.float32)
        # Usuários novos usam um sorteio de desempate próprio (fora da faixa do dataset)
        return self._recommend(R_row, collaborative, n, row_id=self.index.n_users)

    def similar_movies(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        """Filmes com gêneros mais parecidos (cosseno do TF-IDF).

        Empates (muito comuns, pois vários filmes têm os mesmos gêneros) são
        desfeitos pela popularidade, e o próprio filme nunca entra na lista.
        """
        j = self._item_position(movie_id)
        similarity = self.item_features @ self.item_features[j]
        similarity[j] = -np.inf
        order = np.lexsort((-self.popularity, -similarity))[:n]
        result = self.movies.iloc[order][["MovieID", "Title", "Genres"]].copy()
        result["Score"] = similarity[order]
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------
    def _recommend(self, R_row: csr_matrix, collaborative: np.ndarray, n: int, row_id: int) -> pd.DataFrame:
        n = self._check_n(n, R_row)
        seen = R_row.toarray() > 0
        weights = content_weights(R_row, self.config.content_weighted)
        content = np.asarray((np.asarray(weights @ self.item_features)) @ self.item_features.T, dtype=np.float32)
        scores = combine(collaborative, content, R_row, seen, alpha=self.config.alpha, c=self.config.c)
        top = top_k(scores, seen, k=n, row_ids=np.array([row_id]))[0]
        result = self.movies.iloc[top][["MovieID", "Title", "Genres"]].copy()
        result["Score"] = scores[0, top]
        return result.reset_index(drop=True)

    def _ratings_row(self, ratings: dict[int, float]) -> csr_matrix:
        if not ratings:
            raise ValueError("Informe ao menos um filme avaliado.")
        columns, values = [], []
        for movie_id, rating in ratings.items():
            if not 1 <= rating <= 5:
                raise ValueError(f"Nota {rating} inválida para o filme {movie_id}: use 1 a 5.")
            columns.append(self._item_position(movie_id))
            values.append(rating)
        return csr_matrix(
            (np.asarray(values, dtype=np.float32), (np.zeros(len(columns), dtype=int), columns)),
            shape=(1, self.index.n_items),
        )

    def _item_position(self, movie_id: int) -> int:
        if movie_id not in self.index.item_pos.index:
            raise KeyError(f"MovieID {movie_id} não encontrado.")
        return int(self.index.item_pos[movie_id])

    def _check_n(self, n: int, R_row: csr_matrix) -> int:
        if n < 1:
            raise ValueError("`n` deve ser positivo.")
        unseen = self.index.n_items - R_row.nnz
        return min(n, unseen)  # nunca recomenda um filme já visto para completar a lista

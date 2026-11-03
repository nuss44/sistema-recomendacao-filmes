"""Dados sintéticos no formato do MovieLens, para testar sem baixar o dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]


def make_movielens(n_users: int = 300, n_movies: int = 250, mean_ratings: int = 40, seed: int = 0):
    """Gera (movies, ratings) com as mesmas colunas e propriedades relevantes do MovieLens 1M:

    - cada usuário tem ao menos 20 avaliações;
    - popularidade dos filmes concentrada (cauda longa);
    - notas de 1 a 5 com estrutura latente (gostos parecidos entre usuários);
    - vários filmes com exatamente os mesmos gêneros (empates no conteúdo);
    - avaliações do mesmo usuário com o mesmo timestamp (empates no split);
    - alguns filmes que ninguém avaliou;
    - MovieIDs não contíguos e linhas de avaliações fora de ordem.
    """
    rng = np.random.default_rng(seed)

    movie_ids = np.sort(rng.choice(np.arange(1, n_movies * 2), size=n_movies, replace=False))
    genre_sets = []
    for _ in range(n_movies):
        k = rng.integers(1, 4)
        genre_sets.append("|".join(sorted(rng.choice(GENRES[:10], size=k, replace=False))))
    movies = pd.DataFrame({
        "MovieID": movie_ids,
        "Title": [f"Filme {m} ({1950 + m % 50})" for m in movie_ids],
        "Genres": genre_sets,
    })

    # Os últimos 5% dos filmes nunca são avaliados
    rated_movies = movie_ids[: int(n_movies * 0.95)]
    popularity = rng.pareto(1.2, size=len(rated_movies)) + 0.05
    popularity /= popularity.sum()

    user_taste = rng.normal(size=(n_users, 4))
    movie_traits = rng.normal(size=(len(rated_movies), 4))

    rows = []
    for u in range(n_users):
        n = int(np.clip(rng.poisson(mean_ratings), 20, len(rated_movies) - 5))
        chosen = rng.choice(len(rated_movies), size=n, replace=False, p=popularity)
        affinity = movie_traits[chosen] @ user_taste[u] + rng.normal(scale=0.8, size=n)
        stars = np.clip(np.round(3.5 + affinity), 1, 5).astype(int)
        start = rng.integers(956_700_000, 1_040_000_000)
        timestamps = start + np.sort(rng.integers(0, 3_000_000, size=n))
        timestamps[rng.random(n) < 0.3] = start  # vários empates no mesmo instante
        for j, r, t in zip(chosen, stars, timestamps):
            rows.append((u + 1, int(rated_movies[j]), int(r), int(t)))

    ratings = pd.DataFrame(rows, columns=["UserID", "MovieID", "Rating", "Timestamp"])
    ratings = ratings.sample(frac=1, random_state=seed).reset_index(drop=True)
    return movies, ratings

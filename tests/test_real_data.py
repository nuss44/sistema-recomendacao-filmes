"""Testes com o MovieLens 1M real, comparando com valores impressos no notebook.

São pulados automaticamente quando os dados não foram baixados (por exemplo, na
integração contínua). Para rodar: `python scripts/download_data.py` e depois
`pytest tests/test_real_data.py`.
"""

import numpy as np
import pytest

import notebook_reference
from recommender import FINAL_CONFIG, HybridRecommender, Index, load_movielens, temporal_split
from recommender import evaluation as ev
from recommender import models
from recommender.config import DATA_DIR

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "ratings.csv").exists(),
    reason="MovieLens não baixado (rode scripts/download_data.py)",
)


@pytest.fixture(scope="module")
def real():
    movies, ratings = load_movielens()
    return movies, ratings, temporal_split(ratings)


def test_split_sizes_match_notebook(real):
    _, _, split = real
    assert (len(split.train), len(split.validation), len(split.test)) == (640_166, 160_027, 200_016)


def test_production_recommendations_for_user_1_match_notebook(real):
    """Última célula do notebook: `recomendar(1)` com o modelo ajustado em todas as avaliações."""
    movies, ratings, _ = real
    result = HybridRecommender.fit(ratings, movies).recommend_for_user(1, n=10)
    assert list(result["MovieID"]) == [318, 364, 34, 1225, 593, 1704, 1196, 1282, 953, 1302]
    np.testing.assert_allclose(
        result["Score"],
        [0.876331, 0.807021, 0.752195, 0.708167, 0.703288, 0.703195, 0.638210, 0.619919, 0.616898, 0.616377],
        atol=1e-5,
    )


def test_hybrid_and_recommender_match_notebook_on_real_data(real):
    """Mesma verificação do notebook (0 divergências), agora contra o código do pacote."""
    movies, ratings, split = real
    notebook = notebook_reference.load(movies, ratings)
    config = {"n_components": 20, "conteudo_ponderado": True, "alpha": 0.7, "c": None}

    index = Index.from_data(ratings, movies)
    R = index.matrix(split.dev)
    mine = models.hybrid_scores(R, models.genre_features(movies), FINAL_CONFIG)
    theirs = notebook["score_hibrido"](notebook["matriz"](notebook["dev_df"]), config)
    np.testing.assert_array_equal(mine, theirs)

    evaluated = ev.top_k(mine, ev.seen_mask(R))
    recommender = HybridRecommender.fit(split.dev, movies)
    divergences = sum(
        not np.array_equal(
            index.item_pos[recommender.recommend_for_user(uid)["MovieID"]].to_numpy(),
            evaluated[index.user_pos[uid]],
        )
        for uid in index.user_ids
    )
    assert divergences == 0

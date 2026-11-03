"""Comportamento do recomendador servido: usuários novos, filmes similares, erros e persistência."""

import numpy as np
import pytest

from recommender import HybridConfig, HybridRecommender


def ratings_of(rec, user_id):
    """Notas do usuário no formato {MovieID: nota}."""
    row = rec.R[rec.index.user_pos[user_id]]
    return {int(rec.index.item_ids[j]): float(v) for j, v in zip(row.indices, row.data, strict=True)}


def test_new_user_with_same_ratings_gets_same_list(recommender_dev, index):
    """O fold-in de um usuário novo dá o mesmo resultado que o treino deu para um
    usuário com as mesmas notas."""
    for user_id in index.user_ids[:50]:
        new_user = recommender_dev.recommend_for_ratings(ratings_of(recommender_dev, user_id))
        existing = recommender_dev.recommend_for_user(user_id)
        assert list(new_user["MovieID"]) == list(existing["MovieID"])
        np.testing.assert_allclose(new_user["Score"], existing["Score"], atol=1e-5)


def test_new_user_never_gets_movies_already_rated(recommender_dev, movies):
    rated = {int(m): 5 for m in movies["MovieID"].iloc[:5]}
    result = recommender_dev.recommend_for_ratings(rated, n=20)
    assert len(result) == 20
    assert not set(result["MovieID"]) & set(rated)
    assert result["Score"].is_monotonic_decreasing


def test_new_user_with_equal_ratings_still_gets_recommendations(recommender_dev, movies):
    """Notas todas iguais zeram o perfil de conteúdo ponderado; o colaborativo segue funcionando."""
    rated = {int(m): 4 for m in movies["MovieID"].iloc[:3]}
    result = recommender_dev.recommend_for_ratings(rated)
    assert len(result) == 10
    assert result["Score"].notna().all()


def test_new_user_input_is_validated(recommender_dev, movies):
    known = int(movies["MovieID"].iloc[0])
    with pytest.raises(ValueError):
        recommender_dev.recommend_for_ratings({})  # nenhum filme
    with pytest.raises(KeyError):
        recommender_dev.recommend_for_ratings({-1: 5})  # filme inexistente
    for invalid in (0, 6, 4.5 + 2):
        with pytest.raises(ValueError):
            recommender_dev.recommend_for_ratings({known: invalid})  # nota fora de 1–5


def test_unknown_user_raises(recommender_dev):
    with pytest.raises(KeyError):
        recommender_dev.recommend_for_user(10**9)


def test_n_is_capped_by_unseen_movies(recommender_dev, index):
    user_id = index.user_ids[0]
    unseen = index.n_items - recommender_dev.R[index.user_pos[user_id]].nnz
    assert len(recommender_dev.recommend_for_user(user_id, n=10**6)) == unseen
    with pytest.raises(ValueError):
        recommender_dev.recommend_for_user(user_id, n=0)


def test_similar_movies(recommender_dev, movies):
    movie_id = int(movies["MovieID"].iloc[0])
    result = recommender_dev.similar_movies(movie_id, n=15)
    assert len(result) == 15
    assert movie_id not in set(result["MovieID"])
    assert result["Score"].is_monotonic_decreasing
    # Entre filmes com a mesma similaridade, os mais populares vêm primeiro
    popularity = dict(zip(recommender_dev.index.item_ids, recommender_dev.popularity, strict=True))
    for _, tied in result.groupby("Score", sort=False):
        counts = [popularity[m] for m in tied["MovieID"]]
        assert counts == sorted(counts, reverse=True)


def test_save_and_load(recommender_dev, index, tmp_path):
    path = recommender_dev.save(tmp_path / "model.joblib")
    loaded = HybridRecommender.load(path)
    user_id = index.user_ids[0]
    before, after = recommender_dev.recommend_for_user(user_id), loaded.recommend_for_user(user_id)
    assert list(before["MovieID"]) == list(after["MovieID"])
    # Arrays carregados do disco podem seguir outro caminho no BLAS: diferença de ~1e-7.
    np.testing.assert_allclose(before["Score"], after["Score"], rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [{"alpha": None, "c": None}, {"alpha": 0.5, "c": 5}, {"alpha": 1.5}, {"n_components": 0}],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        HybridConfig(**kwargs)


def test_load_rejects_other_objects(tmp_path):
    import joblib

    path = tmp_path / "other.joblib"
    joblib.dump({"not": "a model"}, path)
    with pytest.raises(TypeError):
        HybridRecommender.load(path)

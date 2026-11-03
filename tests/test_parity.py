"""O pacote precisa reproduzir o notebook exatamente: mesmos dados, mesmos scores,
mesmas listas e mesmas métricas. Qualquer diferença aqui invalida os resultados
reportados no README."""

import numpy as np
import pandas as pd
import pytest
from conftest import NOTEBOOK_CONFIG

from recommender import FINAL_CONFIG, models
from recommender import evaluation as ev


def test_split_is_identical(split, notebook):
    pd.testing.assert_frame_equal(split.train, notebook["train_df"])
    pd.testing.assert_frame_equal(split.validation, notebook["val_df"])
    pd.testing.assert_frame_equal(split.test, notebook["test_df"])
    pd.testing.assert_frame_equal(split.dev, notebook["dev_df"])


def test_matrix_is_identical(split, index, notebook):
    mine = index.matrix(split.dev)
    theirs = notebook["matriz"](notebook["dev_df"])
    assert mine.shape == theirs.shape
    assert (mine != theirs).nnz == 0


@pytest.fixture(scope="module")
def dev_data(split, index, notebook):
    R = index.matrix(split.dev)
    return {
        "R": R,
        "R_notebook": notebook["matriz"](notebook["dev_df"]),
        "seen": ev.seen_mask(R),
        "relevant": ev.relevant_items(split.test, index),
    }


def test_relevant_items_are_identical(dev_data, notebook):
    theirs = notebook["relevantes_de"](notebook["test_df"])
    mine = dev_data["relevant"]
    assert list(mine.index) == list(theirs.index)
    for u in mine.index:
        np.testing.assert_array_equal(mine[u], theirs[u])


MODELS = {
    "aleatório": (lambda R, X: models.random_scores(R), lambda nb, R: nb["score_aleatorio"](R)),
    "popularidade": (lambda R, X: models.popularity_scores(R), lambda nb, R: nb["score_popularidade"](R)),
    "svd": (lambda R, X: models.svd_scores(R, 20), lambda nb, R: nb["score_svd_sklearn"](R, 20)),
    "conteúdo": (
        lambda R, X: models.content_scores(R, X, weighted=False),
        lambda nb, R: nb["score_conteudo"](R, ponderado=False),
    ),
    "conteúdo ponderado": (
        lambda R, X: models.content_scores(R, X, weighted=True),
        lambda nb, R: nb["score_conteudo"](R, ponderado=True),
    ),
    "híbrido final": (
        lambda R, X: models.hybrid_scores(R, X, FINAL_CONFIG),
        lambda nb, R: nb["score_hibrido"](R, NOTEBOOK_CONFIG),
    ),
}


@pytest.mark.parametrize(
    "name", MODELS, ids=["aleatorio", "popularidade", "svd", "conteudo", "conteudo-ponderado", "hibrido-final"]
)
def test_scores_lists_and_metrics_are_identical(name, dev_data, item_features, notebook):
    mine_fn, theirs_fn = MODELS[name]
    mine = mine_fn(dev_data["R"], item_features)
    theirs = theirs_fn(notebook, dev_data["R_notebook"])
    seen, relevant = dev_data["seen"], dev_data["relevant"]

    np.testing.assert_array_equal(mine, theirs)
    np.testing.assert_array_equal(ev.top_k(mine, seen), notebook["top_k"](theirs, seen))
    assert ev.evaluate(mine, name, seen, relevant) == notebook["avaliar"](theirs, name, seen, relevant)


def test_unnormalized_hybrid_is_identical(dev_data, item_features, notebook):
    """Híbrido inicial (α = 0,5 sem normalizar), linha de referência na tabela do teste."""
    R, seen = dev_data["R"], dev_data["seen"]
    collaborative = models.svd_scores(R, 100 if R.shape[1] > 100 else 10)
    content = models.content_scores(R, item_features, weighted=False)
    mine = models.combine(collaborative, content, R, seen, alpha=0.5, normalize=False)
    theirs = notebook["combinar"](collaborative, content, R, seen, alpha=0.5, normalizar=False)
    np.testing.assert_array_equal(mine, theirs)


def test_adaptive_alpha_is_identical(dev_data, item_features, notebook):
    R, seen = dev_data["R"], dev_data["seen"]
    collaborative = models.svd_scores(R, 20)
    content = models.content_scores(R, item_features, weighted=True)
    mine = models.combine(collaborative, content, R, seen, c=5)
    theirs = notebook["combinar"](collaborative, content, R, seen, c=5)
    np.testing.assert_array_equal(mine, theirs)


def test_surprise_scores_are_identical(split, index, notebook):
    pytest.importorskip("surprise")
    params = {"n_factors": 10, "lr_all": 0.005, "reg_all": 0.02, "n_epochs": 5}
    mine = models.surprise_scores(split.dev, params, index)
    theirs = notebook["score_surprise"](notebook["dev_df"], params)
    np.testing.assert_array_equal(mine, theirs)


def test_recommender_returns_the_evaluated_lists(recommender_dev, dev_data, item_features, index):
    """A verificação final do notebook: o recomendador servido gera exatamente
    as listas que foram avaliadas no teste."""
    evaluated = ev.top_k(models.hybrid_scores(dev_data["R"], item_features, FINAL_CONFIG), dev_data["seen"])
    for user_id in index.user_ids:
        served = recommender_dev.recommend_for_user(user_id, n=10)
        positions = index.item_pos[served["MovieID"]].to_numpy()
        np.testing.assert_array_equal(positions, evaluated[index.user_pos[user_id]])


def test_recommender_matches_notebook_recommender(recommender_dev, split, movies, notebook, index):
    recommend = notebook["criar_recomendador"](notebook["dev_df"], NOTEBOOK_CONFIG)
    for user_id in index.user_ids:
        mine = recommender_dev.recommend_for_user(user_id, n=10)
        theirs = recommend(user_id, n=10)
        np.testing.assert_array_equal(mine["MovieID"].to_numpy(), theirs["MovieID"].to_numpy())
        # O score de um único usuário pode diferir no último bit do float32 (~1e-7):
        # o BLAS usa rotinas diferentes para uma linha e para a matriz inteira.
        np.testing.assert_allclose(mine["Score"].to_numpy(), theirs["Score"].to_numpy(), rtol=0, atol=1e-6)

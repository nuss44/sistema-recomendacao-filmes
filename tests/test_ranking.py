"""Regra de ranking: maior score primeiro; empates desfeitos por um sorteio fixo de cada usuário."""

import numpy as np
import pytest

from recommender.evaluation import tie_break_keys, top_k


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_without_ties_matches_the_notebook(notebook, rng):
    """Sem empates, a função do pacote dá exatamente o top-k do notebook."""
    # cada linha é uma permutação de 0..299: nenhum empate
    scores = rng.permuted(np.tile(np.arange(300, dtype=np.float32), (200, 1)), axis=1)
    seen = rng.random(scores.shape) < 0.1
    np.testing.assert_array_equal(top_k(scores, seen), notebook["top_k_original"](scores, seen))


def test_ties_are_broken_by_each_users_draw(rng):
    """Com empates, o resultado é definido: maior score, depois o sorteio do usuário."""
    scores = np.round(rng.random((200, 300), dtype=np.float32) * 5).astype(np.float32)
    seen = rng.random(scores.shape) < 0.1
    masked = np.where(seen, -np.inf, scores).astype(np.float32)
    expected = np.stack([np.lexsort((tie_break_keys(u, 300), -row))[:10] for u, row in enumerate(masked)])
    np.testing.assert_array_equal(top_k(scores, seen), expected)


def test_one_user_at_a_time_matches_all_users_at_once(rng):
    """O recomendador calcula um usuário por vez; a avaliação, todos juntos. Os
    desempates precisam coincidir."""
    scores = np.round(rng.random((100, 300), dtype=np.float32) * 3).astype(np.float32)
    seen = rng.random(scores.shape) < 0.1
    together = top_k(scores, seen)
    for u in range(len(scores)):
        alone = top_k(scores[u : u + 1], seen[u : u + 1], row_ids=np.array([u]))[0]
        np.testing.assert_array_equal(alone, together[u])


def test_ties_do_not_favor_the_same_movies_for_everyone():
    """Com tudo empatado, usuários diferentes recebem filmes diferentes (sem vício de catálogo)."""
    scores = np.ones((1000, 500), dtype=np.float32)
    seen = np.zeros_like(scores, dtype=bool)
    chosen = top_k(scores, seen)
    coverage = len(np.unique(chosen)) / scores.shape[1]
    assert coverage > 0.99
    assert np.bincount(chosen.ravel(), minlength=500).max() < 50  # ~20 esperado por filme


def test_seen_and_nan_are_never_recommended(rng):
    scores = rng.random((50, 40), dtype=np.float32)
    seen = rng.random(scores.shape) < 0.3
    scores[rng.random(scores.shape) < 0.1] = np.nan
    top = top_k(scores, seen, k=5)
    rows = np.arange(len(top))[:, None]
    assert not seen[rows, top].any()
    assert not np.isnan(scores[rows, top]).any()


def test_all_ties_follow_the_draw():
    scores = np.ones((3, 20), dtype=np.float32)
    seen = np.zeros_like(scores, dtype=bool)
    seen[:, 0] = True
    top = top_k(scores, seen, k=4)
    for u in range(3):
        keys = tie_break_keys(u, 20)
        keys[0] = np.inf  # visto: nunca entra
        np.testing.assert_array_equal(top[u], np.argsort(keys)[:4])

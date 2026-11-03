import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import notebook_reference  # noqa: E402
from synthetic import make_movielens  # noqa: E402

from recommender import HybridRecommender, Index, temporal_split  # noqa: E402
from recommender.evaluation import top_k  # noqa: E402
from recommender.models import genre_features  # noqa: E402

# Configuração final do notebook, no formato que as funções dele esperam.
NOTEBOOK_CONFIG = {"n_components": 20, "conteudo_ponderado": True, "alpha": 0.7, "c": None}


@pytest.fixture(scope="session")
def dataset():
    return make_movielens(n_users=300, n_movies=250, mean_ratings=40, seed=1)


@pytest.fixture(scope="session")
def movies(dataset):
    return dataset[0]


@pytest.fixture(scope="session")
def ratings(dataset):
    return dataset[1]


@pytest.fixture(scope="session")
def notebook(movies, ratings):
    """Namespace do notebook executado com os dados sintéticos, usando a mesma regra
    de desempate do pacote (veja `recommender.evaluation.top_k`)."""
    return notebook_reference.load(movies, ratings, top_k=top_k)


@pytest.fixture(scope="session")
def split(ratings):
    return temporal_split(ratings)


@pytest.fixture(scope="session")
def index(ratings, movies):
    return Index.from_data(ratings, movies)


@pytest.fixture(scope="session")
def item_features(movies):
    return genre_features(movies)


@pytest.fixture(scope="session")
def recommender_dev(split, movies):
    return HybridRecommender.fit(split.dev, movies)

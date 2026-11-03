"""Títulos legíveis e busca, com títulos no formato real do MovieLens."""

import numpy as np
import pandas as pd
import pytest

from recommender.catalog import display_title, normalize, search, year_of


@pytest.mark.parametrize(
    "title, expected",
    [
        ("Shawshank Redemption, The (1994)", "The Shawshank Redemption (1994)"),
        ("Misérables, Les (1995)", "Les Misérables (1995)"),
        ("Avventura, L' (1960)", "L'Avventura (1960)"),
        ("Haunted World of Edward D. Wood Jr., The (1995)", "The Haunted World of Edward D. Wood Jr. (1995)"),
        (
            "City of Lost Children, The (Cité des enfants perdus, La) (1995)",
            "The City of Lost Children (Cité des enfants perdus, La) (1995)",
        ),
        ("Toy Story (1995)", "Toy Story (1995)"),
        ("Seven (Se7en) (1995)", "Seven (Se7en) (1995)"),
    ],
    ids=["the", "les", "l-apostrofo", "virgula-no-nome", "titulo-alternativo", "sem-artigo", "parenteses"],
)
def test_display_title(title, expected):
    assert display_title(title) == expected


def test_year_and_normalization():
    assert year_of("Toy Story (1995)") == 1995
    assert year_of("Sem ano") is None
    assert normalize("Cité des ENFANTS") == "cite des enfants"


@pytest.fixture
def catalog():
    movies = pd.DataFrame(
        {
            "MovieID": [1, 2, 3, 4],
            "Title": ["Matrix, The (1999)", "Misérables, Les (1995)", "Toy Story (1995)", "Toy Story 2 (1999)"],
            "Genres": ["Action|Sci-Fi", "Drama", "Animation", "Animation"],
        }
    )
    popularity = np.array([10, 5, 8, 9])
    return movies, popularity


@pytest.mark.parametrize(
    "query, expected",
    [
        ("the matrix", [1]),  # título legível
        ("matrix, the", [1]),  # título original
        ("MISERABLES", [2]),  # sem acento e sem diferenciar maiúsculas
        ("toy story", [4, 3]),  # mais popular primeiro
        ("", [1, 4, 3, 2]),  # busca vazia: por popularidade
        ("nada disso", []),
    ],
    ids=["legivel", "original", "acento", "popularidade", "vazia", "sem-resultado"],
)
def test_search(catalog, query, expected):
    movies, popularity = catalog
    assert search(movies, query, popularity)["MovieID"].tolist() == expected

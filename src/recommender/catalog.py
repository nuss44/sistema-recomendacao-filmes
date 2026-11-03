"""Títulos legíveis e busca no catálogo de filmes.

O MovieLens guarda títulos no formato de índice de biblioteca, com o artigo no fim:
"Shawshank Redemption, The (1994)". Para exibir e buscar, o artigo volta para o início.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

# Artigos que o MovieLens move para o fim do título (inglês e línguas dos filmes estrangeiros)
ARTICLES = ("The", "A", "An", "La", "Le", "Les", "L'", "Il", "El", "Los", "Las", "Das", "Der", "Die", "Det", "Den")
_TRAILING_ARTICLE = re.compile(
    r"^(?P<name>.+), (?P<article>" + "|".join(re.escape(a) for a in ARTICLES) + r")(?P<rest> \(.*)?$"
)
_YEAR = re.compile(r"\((\d{4})\)\s*$")


def display_title(title: str) -> str:
    """'Shawshank Redemption, The (1994)' -> 'The Shawshank Redemption (1994)'."""
    match = _TRAILING_ARTICLE.match(title)
    if not match:
        return title
    article = match["article"]
    separator = "" if article.endswith("'") else " "
    return f"{article}{separator}{match['name']}{match['rest'] or ''}"


def year_of(title: str) -> int | None:
    match = _YEAR.search(title)
    return int(match[1]) if match else None


def normalize(text: str) -> str:
    """Minúsculas e sem acentos, para a busca: 'Cité' e 'cite' são iguais."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def search(movies: pd.DataFrame, query: str, popularity: np.ndarray, limit: int = 20) -> pd.DataFrame:
    """Filmes cujo título contém todas as palavras da busca, os mais populares primeiro.

    Procura no título original e no legível, então "the matrix" e "matrix, the" funcionam.
    Busca vazia devolve os mais populares.
    """
    words = normalize(query).split()
    searchable = (movies["Title"] + " " + movies["Title"].map(display_title)).map(normalize)
    mask = np.ones(len(movies), dtype=bool)
    for word in words:
        mask &= searchable.str.contains(word, regex=False).to_numpy()
    positions = np.flatnonzero(mask)
    positions = positions[np.argsort(-popularity[positions], kind="stable")][:limit]
    return movies.iloc[positions]

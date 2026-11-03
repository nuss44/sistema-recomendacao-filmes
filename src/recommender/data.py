"""Leitura dos dados, split cronológico e matriz usuário x filme."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .config import DATA_DIR, TEST_FRAC, VAL_FRAC


def load_movielens(data_dir: Path | str = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lê movies.csv e ratings.csv gerados por `scripts/download_data.py`."""
    data_dir = Path(data_dir)
    missing = [name for name in ("movies.csv", "ratings.csv") if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Arquivos ausentes em {data_dir}: {', '.join(missing)}. "
            "Rode `python scripts/download_data.py` a partir da raiz do projeto."
        )
    movies = pd.read_csv(data_dir / "movies.csv")
    ratings = pd.read_csv(data_dir / "ratings.csv")
    movies["Genres"] = movies["Genres"].fillna("")
    return movies, ratings


# ----------------------------------------------------------------------
# Split cronológico por usuário
# ----------------------------------------------------------------------
def split_last(df: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa as `frac` avaliações mais recentes de cada usuário (mínimo 1).

    Devolve (anteriores, mais recentes). Notebook: `separar_ultimos`.
    """
    df = df.sort_values(["UserID", "Timestamp"])
    position_from_end = df.groupby("UserID").cumcount(ascending=False)
    n = df.groupby("UserID")["MovieID"].transform("size")
    n_cut = np.maximum(1, np.round(n * frac)).astype(int)
    return df[position_from_end >= n_cut], df[position_from_end < n_cut]


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame       # ajuste durante a seleção de modelos
    validation: pd.DataFrame  # escolha de hiperparâmetros
    test: pd.DataFrame        # avaliação final, usada uma única vez
    dev: pd.DataFrame         # treino + validação: re-treino antes do teste


def temporal_split(ratings: pd.DataFrame, test_frac: float = TEST_FRAC, val_frac: float = VAL_FRAC) -> Split:
    """Treino | validação | teste em ordem cronológica dentro de cada usuário."""
    dev, test = split_last(ratings, test_frac)
    train, validation = split_last(dev, val_frac)
    return Split(train=train, validation=validation, test=test, dev=dev)


# ----------------------------------------------------------------------
# Índices de usuários e filmes
# ----------------------------------------------------------------------
class Index:
    """Mapeia UserID/MovieID para linhas/colunas da matriz de notas.

    Os filmes ficam na mesma ordem de `movies` (inclusive os que nunca foram
    avaliados), como no notebook. A ordem importa: ela define as colunas da
    matriz e, portanto, o resultado do SVD.
    """

    def __init__(self, user_ids: np.ndarray, item_ids: np.ndarray):
        self.user_ids = np.asarray(user_ids)
        self.item_ids = np.asarray(item_ids)
        self.user_pos = pd.Series(np.arange(len(self.user_ids)), index=self.user_ids)
        self.item_pos = pd.Series(np.arange(len(self.item_ids)), index=self.item_ids)

    @classmethod
    def from_data(cls, ratings: pd.DataFrame, movies: pd.DataFrame) -> "Index":
        return cls(np.sort(ratings["UserID"].unique()), movies["MovieID"].to_numpy())

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def matrix(self, df: pd.DataFrame) -> csr_matrix:
        """Matriz esparsa usuário x filme com as notas de `df` (0 = não avaliado).

        Notebook: `matriz`.
        """
        return csr_matrix(
            (
                df["Rating"].astype(np.float32).to_numpy(),
                (self.user_pos[df["UserID"]].to_numpy(), self.item_pos[df["MovieID"]].to_numpy()),
            ),
            shape=(self.n_users, self.n_items),
        )

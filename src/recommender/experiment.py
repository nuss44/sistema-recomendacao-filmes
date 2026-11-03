"""Protocolo de avaliação do notebook: seleção na validação e avaliação única no teste.

1. Fase de validação: modelos ajustados no treino e comparados na validação. Aqui se
   escolhem o número de componentes do SVD, a variante de conteúdo e o peso α.
2. Fase de teste: com tudo escolhido, os modelos são re-treinados no desenvolvimento
   (treino + validação) e avaliados uma única vez no teste.

Notebook: células "Fase de validação", "Híbrido na validação", "Configuração
escolhida" e "Avaliação final no teste".
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from . import models
from .config import RANDOM_STATE, HybridConfig
from .data import Index, Split
from .evaluation import evaluate, relevant_items, seen_mask

SVD_COMPONENTS = (20, 50, 100, 150, 200)
ALPHAS = tuple(float(a) for a in np.round(np.arange(0, 1.01, 0.1), 1))
ADAPTIVE_CS = (5, 10, 15, 25, 50, 100)
CONTENT_NAMES = {False: "Conteúdo original", True: "Conteúdo ponderado por nota"}

# Busca do SVD do Surprise (RMSE na validação) e o resultado obtido no notebook.
SURPRISE_GRID = {
    "n_factors": [50, 100, 150],
    "lr_all": [0.002, 0.005, 0.01],
    "reg_all": [0.02, 0.05, 0.1],
    "n_epochs": [20, 50, 70],
}
NOTEBOOK_SURPRISE_PARAMS = {"n_factors": 150, "lr_all": 0.01, "reg_all": 0.1, "n_epochs": 70}


@dataclass
class Selection:
    """Tudo o que foi decidido na validação."""

    validation: pd.DataFrame     # modelos individuais na validação
    hybrid_search: pd.DataFrame  # todas as combinações do híbrido testadas
    n_components: int            # componentes do SVD escolhidos pelo NDCG
    config: HybridConfig         # configuração final do híbrido
    ndcg_hybrid: float           # NDCG do híbrido escolhido, na validação
    ndcg_svd: float              # NDCG do melhor SVD sozinho, na validação


def _ndcg(k: int) -> str:
    return f"NDCG@{k}"


def select_on_validation(split: Split, index: Index, item_features: np.ndarray, k: int = 10) -> Selection:
    """Fase de validação: escolhe componentes do SVD, variante de conteúdo e α."""
    R = index.matrix(split.train)
    seen = seen_mask(R)
    relevant = relevant_items(split.validation, index)

    rows = [
        evaluate(models.random_scores(R), "Aleatório", seen, relevant, k),
        evaluate(models.popularity_scores(R), "Popularidade", seen, relevant, k),
    ]

    # Componentes do SVD escolhidos pela métrica de ranking (não pelo RMSE).
    # Só a matriz do melhor fica em memória.
    best_svd, best_ndcg, n_components = None, -np.inf, None
    for n in SVD_COMPONENTS:
        scores = models.svd_scores(R, n)
        row = evaluate(scores, f"SVD sklearn ({n} comp.)", seen, relevant, k)
        rows.append(row)
        if row[_ndcg(k)] > best_ndcg:  # ">" mantém o primeiro em caso de empate, como idxmax
            best_svd, best_ndcg, n_components = scores, row[_ndcg(k)], n

    content = {weighted: models.content_scores(R, item_features, weighted) for weighted in (False, True)}
    rows.append(evaluate(content[False], "Conteúdo (original)", seen, relevant, k))
    rows.append(evaluate(content[True], "Conteúdo (ponderado por nota)", seen, relevant, k))
    validation = pd.DataFrame(rows)

    # Híbrido: α fixo de 0 a 1 e α adaptativo n/(n+c), para as duas variantes de conteúdo
    search = []
    for weighted, name in CONTENT_NAMES.items():
        for alpha in ALPHAS:
            scores = models.combine(best_svd, content[weighted], R, seen, alpha=alpha)
            search.append({**evaluate(scores, name, seen, relevant, k),
                           "tipo": "fixo", "alpha": alpha, "c": None, "ponderado": weighted})
        for c in ADAPTIVE_CS:
            scores = models.combine(best_svd, content[weighted], R, seen, c=c)
            search.append({**evaluate(scores, name, seen, relevant, k),
                           "tipo": "adaptativo", "alpha": None, "c": c, "ponderado": weighted})
    hybrid_search = pd.DataFrame(search)

    best = hybrid_search.loc[hybrid_search[_ndcg(k)].idxmax()]
    config = HybridConfig(
        n_components=n_components,
        content_weighted=bool(best["ponderado"]),
        alpha=None if best["tipo"] == "adaptativo" else float(best["alpha"]),
        c=None if best["tipo"] == "fixo" else int(best["c"]),
    )
    return Selection(validation, hybrid_search, n_components, config, float(best[_ndcg(k)]), float(best_ndcg))


def tune_surprise(split: Split, grid: dict = SURPRISE_GRID, n_jobs: int = -1) -> tuple[dict, pd.DataFrame]:
    """Busca em grade do SVD do Surprise: ajuste no treino, menor RMSE na validação.

    Notebook: célula "SVD com Surprise". São 81 combinações; pode levar bastante tempo.
    """
    from joblib import Parallel, delayed
    from surprise import SVD, Dataset, Reader, accuracy

    trainset = Dataset.load_from_df(split.train[["UserID", "MovieID", "Rating"]], Reader(rating_scale=(1, 5))).build_full_trainset()
    validation = list(split.validation[["UserID", "MovieID", "Rating"]].itertuples(index=False, name=None))
    combinations = [dict(zip(grid, values)) for values in product(*grid.values())]

    def rmse(params):
        model = SVD(**params, random_state=RANDOM_STATE)
        model.fit(trainset)
        return accuracy.rmse(model.test(validation), verbose=False)

    rmses = Parallel(n_jobs=n_jobs)(delayed(rmse)(p) for p in combinations)
    table = (pd.DataFrame(combinations).assign(**{"RMSE validação": rmses})
             .sort_values("RMSE validação").reset_index(drop=True))
    best = {key: (int(value) if key in ("n_factors", "n_epochs") else float(value))
            for key, value in table.iloc[0].drop("RMSE validação").items()}
    return best, table


def evaluate_on_test(
    split: Split,
    index: Index,
    item_features: np.ndarray,
    config: HybridConfig,
    surprise_params: dict | None = None,
    k: int = 10,
) -> pd.DataFrame:
    """Fase de teste: re-treino no desenvolvimento e avaliação única no teste.

    Nenhuma escolha é feita aqui. Se `surprise_params` for None (ou o Surprise não
    estiver instalado), a linha do Surprise sai com métricas vazias, sem inventar números.
    """
    R = index.matrix(split.dev)
    seen = seen_mask(R)
    relevant = relevant_items(split.test, index)

    def row(scores, name):
        return evaluate(scores, name, seen, relevant, k)

    rows = [
        row(models.random_scores(R), "Aleatório"),
        row(models.popularity_scores(R), "Popularidade"),
        row(models.svd_scores(R, config.n_components), f"SVD sklearn ({config.n_components} comp.)"),
    ]
    surprise_name = "SVD Surprise (nota prevista)"
    if surprise_params is not None:
        rows.append(row(models.surprise_scores(split.dev, surprise_params, index), surprise_name))
    else:
        rows.append({"Modelo": surprise_name})

    content_original = models.content_scores(R, item_features, weighted=False)
    rows.append(row(content_original, "Conteúdo (original)"))
    rows.append(row(models.content_scores(R, item_features, weighted=True), "Conteúdo (ponderado por nota)"))
    # Versão inicial do notebook, como referência: 100 componentes, conteúdo original, sem normalizar
    initial = models.combine(models.svd_scores(R, 100), content_original, R, seen, alpha=0.5, normalize=False)
    rows.append(row(initial, "Híbrido inicial (α=0.5, sem normalizar)"))
    rows.append(row(models.hybrid_scores(R, item_features, config), "Híbrido final (config. da validação)"))

    table = pd.DataFrame(rows)
    return table.sort_values(_ndcg(k), ascending=False, na_position="last").reset_index(drop=True)

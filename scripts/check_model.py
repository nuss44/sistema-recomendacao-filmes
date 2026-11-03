"""Confere se o modelo treinado é o mesmo do notebook.

Uso (depois de `python scripts/train.py`):
    python scripts/check_model.py

Compara as recomendações do usuário 1 com as impressas na última célula do notebook
(modelo final ajustado com todas as avaliações do MovieLens 1M). Serve para verificar
que outra máquina, outra versão das bibliotecas ou o build do Docker chegaram ao mesmo
modelo. Termina com código 1 se a lista for diferente.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from recommender import HybridRecommender
from recommender.config import MODEL_PATH

# Última célula do notebook: recomendar(1)
EXPECTED_USER = 1
EXPECTED_MOVIES = [318, 364, 34, 1225, 593, 1704, 1196, 1282, 953, 1302]
EXPECTED_SCORES = [0.876331, 0.807021, 0.752195, 0.708167, 0.703288, 0.703195, 0.638210, 0.619919, 0.616898, 0.616377]
TOLERANCE = 1e-5  # arredondamento do float32 entre bibliotecas e processadores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="arquivo do modelo (.joblib)")
    args = parser.parse_args(argv)

    if not args.model.exists():
        print(f"[erro] Modelo não encontrado em {args.model}. Rode `python scripts/train.py`.", file=sys.stderr)
        return 1

    result = HybridRecommender.load(args.model).recommend_for_user(EXPECTED_USER, n=len(EXPECTED_MOVIES))
    movies, scores = result["MovieID"].tolist(), result["Score"].to_numpy()

    if movies != EXPECTED_MOVIES:
        print("[erro] As recomendações do usuário 1 não batem com o notebook.", file=sys.stderr)
        print(f"       esperado: {EXPECTED_MOVIES}", file=sys.stderr)
        print(f"       obtido:   {movies}", file=sys.stderr)
        return 1
    difference = float(np.abs(scores - np.asarray(EXPECTED_SCORES)).max())
    if difference > TOLERANCE:
        print(f"[erro] Mesma lista, mas os scores diferem em até {difference:.2e}.", file=sys.stderr)
        return 1

    print(f"[ok] Modelo idêntico ao do notebook (usuário 1: mesma lista, scores a menos de {difference:.0e}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

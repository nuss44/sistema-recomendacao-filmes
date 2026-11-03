"""Treina o recomendador final com todas as avaliações e salva em models/.

Uso (a partir da raiz do projeto, depois de `python scripts/download_data.py`):
    python scripts/train.py
    python scripts/train.py --data-dir outra/pasta --output outro/modelo.joblib

Saída:
    models/recommender.joblib   o modelo que a API e o Streamlit carregam
    models/metadata.json        configuração, dados e versões usados no treino

A configuração é FINAL_CONFIG, a mesma escolhida na validação e avaliada no teste
(confira com `python scripts/evaluate.py`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from recommender import FINAL_CONFIG, HybridRecommender, load_movielens
from recommender.config import DATA_DIR, MODEL_PATH
from recommender.data import fingerprint
from recommender.reports import environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="pasta com movies.csv e ratings.csv")
    parser.add_argument("--output", type=Path, default=MODEL_PATH, help="arquivo do modelo (.joblib)")
    args = parser.parse_args(argv)
    model_path = args.output

    movies, ratings = load_movielens(args.data_dir)
    count = f"{len(ratings):,}".replace(",", ".")
    print(f"[..] Treinando com {count} avaliações e {FINAL_CONFIG}")
    start = time.perf_counter()
    recommender = HybridRecommender.fit(ratings, movies, FINAL_CONFIG)
    seconds = time.perf_counter() - start

    recommender.save(model_path)
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(FINAL_CONFIG),
        "data": {
            "ratings": len(ratings),
            "users": recommender.index.n_users,
            "movies": recommender.index.n_items,
            "fingerprint": fingerprint(ratings),
        },
        "training_seconds": round(seconds, 2),
        "environment": environment(),
    }
    metadata_path = model_path.with_name("metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    size_mb = model_path.stat().st_size / 1e6
    print(
        f"[ok] {model_path.parent.name}/{model_path.name} ({size_mb:.1f} MB) e {metadata_path.name} em {seconds:.1f}s"
    )

    example_user = int(recommender.index.user_ids[0])
    print(f"\nExemplo — recomendações para o usuário {example_user}:")
    print(recommender.recommend_for_user(example_user, n=5).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

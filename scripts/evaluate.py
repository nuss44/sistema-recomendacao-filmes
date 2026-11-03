"""Refaz a avaliação do notebook e grava as métricas em reports/.

Uso (a partir da raiz do projeto, depois de `python scripts/download_data.py`):
    python scripts/evaluate.py                  # protocolo completo
    python scripts/evaluate.py --skip-surprise  # sem a linha do SVD do Surprise
    python scripts/evaluate.py --tune-surprise  # refaz também a busca do Surprise (lento)
    python scripts/evaluate.py --data-dir outra/pasta --output-dir outra/saida

Saída:
    reports/metrics.json             fonte única das métricas (o README lê daqui)
    reports/metrics.md               as mesmas tabelas em Markdown
    reports/figures/validation_alpha.png
    reports/figures/test_ndcg.png

Termina com código 1 se a configuração escolhida na validação for diferente da
configuração final usada pelo recomendador (FINAL_CONFIG).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

from recommender import FINAL_CONFIG, Index, load_movielens, reports, temporal_split
from recommender.config import DATA_DIR, RELEVANCE_THRESHOLD, REPORTS_DIR, TEST_FRAC, VAL_FRAC, K
from recommender.data import fingerprint
from recommender.experiment import NOTEBOOK_SURPRISE_PARAMS, evaluate_on_test, select_on_validation, tune_surprise
from recommender.models import genre_features


def surprise_available() -> bool:
    try:
        import surprise  # noqa: F401
    except ImportError:
        return False
    return True


def step(message: str) -> float:
    print(f"[..] {message}", flush=True)
    return time.perf_counter()


def done(start: float) -> None:
    print(f"[ok] concluído em {time.perf_counter() - start:.0f}s", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    surprise = parser.add_mutually_exclusive_group()
    surprise.add_argument("--skip-surprise", action="store_true", help="não avalia o SVD do Surprise")
    surprise.add_argument(
        "--tune-surprise", action="store_true", help="refaz a busca em grade do Surprise (81 combinações)"
    )
    parser.add_argument("--no-figures", action="store_true", help="não gera os gráficos (dispensa o matplotlib)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="pasta com movies.csv e ratings.csv")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR, help="pasta dos relatórios")
    args = parser.parse_args(argv)
    output = args.output_dir

    movies, ratings = load_movielens(args.data_dir)
    split = temporal_split(ratings)
    index = Index.from_data(ratings, movies)
    item_features = genre_features(movies)
    fmt = lambda n: f"{n:,}".replace(",", ".")  # noqa: E731
    print(
        f"[ok] {fmt(len(ratings))} avaliações · treino {fmt(len(split.train))} · validação "
        f"{fmt(len(split.validation))} · teste {fmt(len(split.test))}"
    )

    # ---------------- Fase de validação ----------------
    start = step("Fase de validação: SVD, conteúdo e busca do híbrido (34 combinações)")
    selection = select_on_validation(split, index, item_features)
    done(start)
    print(f"[ok] Configuração escolhida: {selection.config}")
    print(f"     NDCG@{K} na validação: híbrido {selection.ndcg_hybrid:.4f} | SVD sozinho {selection.ndcg_svd:.4f}")

    # ---------------- Parâmetros do Surprise ----------------
    surprise_params, surprise_source, surprise_search = None, None, None
    if not args.skip_surprise:
        if not surprise_available():
            print(
                '[!!] scikit-surprise não instalado: linha do Surprise fica "não executado". '
                'Instale com: python -m pip install -e ".[experiments]"'
            )
        elif args.tune_surprise:
            start = step("Busca em grade do Surprise na validação (81 combinações, pode demorar)")
            surprise_params, surprise_search = tune_surprise(split)
            surprise_source = "busca refeita por scripts/evaluate.py --tune-surprise"
            done(start)
        else:
            surprise_params = NOTEBOOK_SURPRISE_PARAMS
            surprise_source = "busca do notebook (RMSE na validação); refaça com --tune-surprise"

    # ---------------- Fase de teste ----------------
    start = step("Fase de teste: re-treino em treino + validação, avaliação única no teste")
    test = evaluate_on_test(split, index, item_features, selection.config, surprise_params)
    done(start)

    # ---------------- Relatórios ----------------
    matches = selection.config == FINAL_CONFIG
    payload = {
        "environment": reports.environment(),
        "data": {
            "ratings": len(ratings),
            "users": ratings["UserID"].nunique(),
            "movies": len(movies),
            "fingerprint": fingerprint(ratings),
        },
        "protocol": {
            "split": "cronológico por usuário",
            "test_frac": TEST_FRAC,
            "val_frac": VAL_FRAC,
            "k": K,
            "relevance_threshold": RELEVANCE_THRESHOLD,
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "selection": {
            "config": asdict(selection.config),
            "matches_final_config": matches,
            "ndcg_validation_hybrid": selection.ndcg_hybrid,
            "ndcg_validation_svd": selection.ndcg_svd,
            "surprise_params": surprise_params,
            "surprise_params_source": surprise_source,
        },
        "test": reports.records(test),
        "validation": reports.records(selection.validation),
        "hybrid_search": reports.records(selection.hybrid_search),
    }
    if surprise_search is not None:
        payload["surprise_search"] = reports.records(surprise_search)

    output.mkdir(parents=True, exist_ok=True)
    reports.write_json(payload, output / "metrics.json")
    final = test.loc[test["Modelo"].str.startswith("Híbrido final")].iloc[0]
    popularity = test.loc[test["Modelo"] == "Popularidade"].iloc[0]
    summary = [
        f"- Configuração escolhida na validação: SVD com {selection.config.n_components} componentes, "
        f"conteúdo {'ponderado por nota' if selection.config.content_weighted else 'original'}, "
        + (
            f"α = {selection.config.alpha:.1f}".replace(".", ",")
            if selection.config.alpha is not None
            else f"α adaptativo (c = {selection.config.c})"
        ),
        (
            f"- Híbrido final no teste: NDCG@{K} = {final[f'NDCG@{K}']:.4f}, cobertura = {final['Cobertura']:.1%} "
            f"(popularidade: {popularity[f'NDCG@{K}']:.4f}, {popularity['Cobertura']:.1%})"
        ).replace(".", ","),
        f"- Dados: {fmt(len(ratings))} avaliações, impressão digital `{fingerprint(ratings)}`",
    ]
    reports.write_markdown(selection.validation, test, summary, output / "metrics.md")
    print(f"[ok] {output.name}/metrics.json e {output.name}/metrics.md")

    if not args.no_figures:
        try:
            figures = output / "figures"
            reports.plot_alpha_search(selection.hybrid_search, selection.validation, figures / "validation_alpha.png")
            reports.plot_test_ndcg(test, figures / "test_ndcg.png")
            print(f"[ok] {output.name}/figures/validation_alpha.png e test_ndcg.png")
        except ImportError:
            print('[!!] matplotlib não instalado: gráficos não gerados (python -m pip install -e ".[dev]")')

    print("\n" + reports.markdown_table(test))
    if not matches:
        print(
            f"\n[erro] A validação escolheu {selection.config}, diferente de FINAL_CONFIG = {FINAL_CONFIG}.",
            file=sys.stderr,
        )
        print("       O recomendador servido não corresponde ao modelo avaliado. Revise config.py.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

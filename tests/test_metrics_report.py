"""reports/metrics.json precisa bater com as tabelas impressas no notebook.

Este teste não precisa dos dados: ele lê o relatório versionado no repositório.
Assim, qualquer pessoa (e a integração contínua) consegue conferir que as métricas
publicadas são as do experimento. Pulado enquanto o relatório não foi gerado.
"""

import json

import pytest

from recommender import FINAL_CONFIG
from recommender.config import REPORTS_DIR

METRICS = REPORTS_DIR / "metrics.json"
pytestmark = pytest.mark.skipif(not METRICS.exists(), reason="rode scripts/evaluate.py para gerar o relatório")

COLUMNS = ("Precision@10", "Recall@10", "NDCG@10", "Cobertura")

# Modelos de conteúdo puro: filmes com os mesmos gêneros recebem o mesmo score, e todo
# usuário tem empate na fronteira do top-10. O notebook desempatava de forma arbitrária
# (muda com o NumPy e o processador), então os valores dele para estas linhas não são
# reprodutíveis e não servem de referência. O pacote desempata pela ordem do catálogo;
# estas linhas são conferidas contra um recálculo em tests/test_real_data.py.
TIE_DEPENDENT = {"Conteúdo (original)", "Conteúdo (ponderado por nota)"}

# Saída da célula "Avaliação final no teste" do notebook (tabela_teste.round(4))
NOTEBOOK_TEST = {
    "SVD sklearn (20 comp.)": (0.0992, 0.0847, 0.1218, 0.1774),
    "Híbrido final (config. da validação)": (0.0997, 0.0844, 0.1213, 0.2189),
    "Popularidade": (0.0798, 0.0472, 0.0907, 0.0301),
    "Híbrido inicial (α=0.5, sem normalizar)": (0.0381, 0.0400, 0.0520, 0.4569),
    "SVD Surprise (nota prevista)": (0.0243, 0.0168, 0.0252, 0.0860),
    "Conteúdo (ponderado por nota)": (0.0122, 0.0095, 0.0144, 0.9191),
    "Conteúdo (original)": (0.0098, 0.0101, 0.0125, 0.7744),
    "Aleatório": (0.0051, 0.0029, 0.0054, 1.0000),
}

# Saída da célula "Fase de validação" do notebook (tabela_val.round(4))
NOTEBOOK_VALIDATION = {
    "Aleatório": (0.0038, 0.0025, 0.0042, 1.0000),
    "Popularidade": (0.0676, 0.0520, 0.0796, 0.0252),
    "SVD sklearn (20 comp.)": (0.0801, 0.0882, 0.1027, 0.1617),
    "SVD sklearn (50 comp.)": (0.0761, 0.0905, 0.0984, 0.2163),
    "SVD sklearn (100 comp.)": (0.0714, 0.0891, 0.0941, 0.2640),
    "SVD sklearn (150 comp.)": (0.0663, 0.0835, 0.0883, 0.2933),
    "SVD sklearn (200 comp.)": (0.0615, 0.0783, 0.0838, 0.3235),
    "Conteúdo (original)": (0.0102, 0.0115, 0.0130, 0.7819),
    "Conteúdo (ponderado por nota)": (0.0108, 0.0088, 0.0125, 0.9122),
}


@pytest.fixture(scope="module")
def metrics():
    return json.loads(METRICS.read_text(encoding="utf-8"))


def assert_table_matches(rows: list[dict], expected: dict) -> None:
    by_name = {row["Modelo"]: row for row in rows}
    assert set(by_name) == set(expected)
    for name, values in expected.items():
        row = by_name[name]
        if row["NDCG@10"] is None:  # linha não executada (Surprise ausente)
            assert name == "SVD Surprise (nota prevista)"
            continue
        if name in TIE_DEPENDENT:
            assert all(0 <= row[column] <= 1 for column in COLUMNS), name
            continue
        for column, value in zip(COLUMNS, values):
            assert row[column] == pytest.approx(value, abs=1e-4), f"{name} · {column}"


def test_report_uses_full_movielens(metrics):
    assert metrics["data"]["ratings"] == 1_000_209
    assert (metrics["protocol"]["train"], metrics["protocol"]["validation"], metrics["protocol"]["test"]) == (
        640_166, 160_027, 200_016)


def test_selection_matches_notebook_and_final_config(metrics):
    selection = metrics["selection"]
    assert selection["matches_final_config"] is True
    assert selection["config"] == {"n_components": 20, "content_weighted": True, "alpha": 0.7, "c": None}
    assert selection["config"]["n_components"] == FINAL_CONFIG.n_components
    assert selection["ndcg_validation_hybrid"] == pytest.approx(0.1073, abs=1e-4)
    assert selection["ndcg_validation_svd"] == pytest.approx(0.1027, abs=1e-4)


def test_validation_table_matches_notebook(metrics):
    assert_table_matches(metrics["validation"], NOTEBOOK_VALIDATION)


def test_test_table_matches_notebook(metrics):
    assert_table_matches(metrics["test"], NOTEBOOK_TEST)

"""O protocolo de avaliação do pacote precisa reproduzir as células de validação e
teste do notebook: mesmas tabelas, mesmas escolhas, mesma tabela final."""

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

import notebook_reference
from recommender import evaluation as ev
from recommender import reports
from recommender.experiment import evaluate_on_test, select_on_validation

SMALL_SURPRISE = {"n_factors": 10, "lr_all": 0.005, "reg_all": 0.02, "n_epochs": 5}

try:
    import surprise  # noqa: F401

    HAS_SURPRISE = True
except ImportError:
    HAS_SURPRISE = False


@pytest.fixture(scope="module")
def protocol(movies, ratings, split, index, item_features):
    pytest.importorskip("matplotlib")  # as células do notebook desenham gráficos
    namespace = notebook_reference.load(movies, ratings, top_k=ev.top_k)
    if not HAS_SURPRISE:
        # Sem o Surprise, a linha dele não é comparada; um placeholder deixa a célula rodar.
        namespace["score_surprise"] = lambda df, params: np.zeros(
            (namespace["n_users"], namespace["n_items"]), np.float32)
    namespace = notebook_reference.run_protocol(namespace, surprise_params=SMALL_SURPRISE)
    selection = select_on_validation(split, index, item_features)
    test = evaluate_on_test(split, index, item_features, selection.config,
                            SMALL_SURPRISE if HAS_SURPRISE else None)
    return namespace, selection, test


def test_validation_table_is_identical(protocol):
    notebook, selection, _ = protocol
    pd.testing.assert_frame_equal(selection.validation, notebook["tabela_val"])


def test_hybrid_search_is_identical(protocol):
    notebook, selection, _ = protocol
    pd.testing.assert_frame_equal(selection.hybrid_search, notebook["busca_hibrido"])


def test_selected_configuration_is_identical(protocol):
    notebook, selection, _ = protocol
    assert selection.n_components == notebook["n_comp_top"]
    config = asdict(selection.config)
    config["conteudo_ponderado"] = config.pop("content_weighted")
    assert config == notebook["config_final"]
    assert selection.ndcg_hybrid == notebook["ndcg_val_escolhido"]
    assert selection.ndcg_svd == notebook["ndcg_val_svd"]


def test_test_table_is_identical(protocol):
    notebook, _, test = protocol
    theirs = notebook["tabela_teste"].set_index("Modelo")
    mine = test.set_index("Modelo")
    if not HAS_SURPRISE:
        name = "SVD Surprise (nota prevista)"
        assert mine.loc[name].isna().all()  # "não executado", nunca um número inventado
        mine, theirs = mine.drop(name), theirs.drop(name)
    pd.testing.assert_frame_equal(mine.sort_index(), theirs.sort_index())


def test_test_table_is_sorted_by_ndcg(protocol):
    _, _, test = protocol
    ndcg = test["NDCG@10"].dropna()
    assert ndcg.is_monotonic_decreasing
    assert test["NDCG@10"].isna().sum() == (0 if HAS_SURPRISE else 1)


# ----------------------------------------------------------------------
# Relatórios
# ----------------------------------------------------------------------
def test_json_report_turns_nan_into_null(protocol, tmp_path):
    _, _, test = protocol
    table = pd.concat([test, pd.DataFrame([{"Modelo": "Teste NaN"}])], ignore_index=True)
    path = reports.write_json({"test": reports.records(table)}, tmp_path / "metrics.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["test"][-1]["NDCG@10"] is None
    assert "generated_at" in data


def test_markdown_table_uses_decimal_comma(protocol):
    _, _, test = protocol
    text = reports.markdown_table(pd.concat([test, pd.DataFrame([{"Modelo": "X"}])], ignore_index=True))
    assert "não executado" in text
    first_value = text.splitlines()[2].split("|")[2].strip()
    assert "," in first_value and "." not in first_value


def test_figures_are_written(protocol, tmp_path):
    pytest.importorskip("matplotlib")
    _, selection, test = protocol
    paths = [
        reports.plot_alpha_search(selection.hybrid_search, selection.validation, tmp_path / "alpha.png"),
        reports.plot_test_ndcg(test, tmp_path / "test.png"),
    ]
    for path in paths:
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

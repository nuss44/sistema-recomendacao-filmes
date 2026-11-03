"""Os três scripts rodando de ponta a ponta, com dados sintéticos em pastas temporárias."""

import importlib.util
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from synthetic import make_movielens

from recommender import FINAL_CONFIG, HybridRecommender
from recommender.data import fingerprint

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------
# download_data.py
# ----------------------------------------------------------------------
@pytest.fixture
def fake_zip(tmp_path):
    """Um ml-1m.zip em miniatura, no formato original: separador '::' e latin-1."""
    folder = tmp_path / "zip" / "ml-1m"
    folder.mkdir(parents=True)
    (folder / "movies.dat").write_bytes(
        "1::Toy Story (1995)::Animation|Children's|Comedy\n"
        "2::Cité des enfants perdus, La (1995)::Adventure|Sci-Fi\n"
        "3::Shawshank Redemption, The (1994)::Drama\n".encode("latin-1")
    )
    (folder / "users.dat").write_text("1::F::1::10::48067\n2::M::56::16::02460\n3::M::25::15::98107-2117\n")
    (folder / "ratings.dat").write_text(
        "1::1::5::978300760\n1::2::3::978302109\n2::3::4::978301968\n3::1::1::978300275\n"
    )
    path = tmp_path / "ml-1m.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for file in folder.iterdir():
            archive.write(file, f"ml-1m/{file.name}")
    return path


def test_download_rejects_data_that_is_not_movielens_1m(fake_zip, tmp_path):
    download = load_script("download_data")
    data_dir = tmp_path / "data"
    assert download.main(["--url", fake_zip.as_uri(), "--data-dir", str(data_dir)]) == 1
    assert (data_dir / "raw" / "ml-1m.zip").exists()
    assert not (data_dir / "processed").exists()  # nada é gravado se a validação falha


def test_download_converts_and_preserves_text(fake_zip, tmp_path, monkeypatch):
    download = load_script("download_data")
    monkeypatch.setattr(download, "EXPECTED_ROWS", {"movies": 3, "users": 3, "ratings": 4})
    data_dir = tmp_path / "data"
    assert download.main(["--url", fake_zip.as_uri(), "--data-dir", str(data_dir)]) == 0

    movies = pd.read_csv(data_dir / "processed" / "movies.csv")
    users = pd.read_csv(data_dir / "processed" / "users.csv", dtype={"Zip-code": str})
    assert movies["Title"].iloc[1] == "Cité des enfants perdus, La (1995)"  # latin-1 -> utf-8
    assert users["Zip-code"].tolist() == ["48067", "02460", "98107-2117"]  # zeros à esquerda
    ratings = pd.read_csv(data_dir / "processed" / "ratings.csv")
    assert list(ratings.columns) == ["UserID", "MovieID", "Rating", "Timestamp"]

    # Segunda execução: nada a fazer
    assert download.main(["--data-dir", str(data_dir)]) == 0


def test_download_reports_a_broken_url(tmp_path):
    download = load_script("download_data")
    missing = (tmp_path / "nao-existe.zip").as_uri()
    assert download.main(["--url", missing, "--data-dir", str(tmp_path / "data")]) == 1


def test_validation_catches_inconsistent_data():
    download = load_script("download_data")
    movies = pd.DataFrame({"MovieID": [1, 1], "Title": ["A", "B"], "Genres": ["Drama", "Drama"]})
    users = pd.DataFrame({"UserID": [1], "Gender": ["F"], "Age": [1], "Occupation": [1], "Zip-code": ["1"]})
    ratings = pd.DataFrame({"UserID": [1, 2], "MovieID": [1, 9], "Rating": [7, 3], "Timestamp": [1, 2]})
    problems = " | ".join(download.validate({"movies": movies, "users": users, "ratings": ratings}))
    for expected in ("MovieID duplicado", "fora do intervalo", "UserID sem cadastro", "MovieID sem cadastro", "linhas"):
        assert expected in problems


# ----------------------------------------------------------------------
# evaluate.py e train.py
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("processed")
    movies, ratings = make_movielens(n_users=300, n_movies=250, mean_ratings=40, seed=1)
    movies.to_csv(folder / "movies.csv", index=False)
    ratings.to_csv(folder / "ratings.csv", index=False)
    return folder


def test_evaluate_writes_reports(data_dir, tmp_path):
    pytest.importorskip("matplotlib")
    evaluate = load_script("evaluate")
    output = tmp_path / "reports"
    code = evaluate.main(["--data-dir", str(data_dir), "--output-dir", str(output), "--skip-surprise"])

    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    # Sai com erro exatamente quando a validação escolhe algo diferente do modelo servido
    assert code == (0 if metrics["selection"]["matches_final_config"] else 1)
    assert metrics["data"]["fingerprint"] == fingerprint(pd.read_csv(data_dir / "ratings.csv"))
    assert {row["Modelo"] for row in metrics["test"]} >= {"Popularidade", "Híbrido final (config. da validação)"}
    surprise = next(row for row in metrics["test"] if row["Modelo"].startswith("SVD Surprise"))
    assert surprise["NDCG@10"] is None  # --skip-surprise: "não executado", nunca um número
    assert "não executado" in (output / "metrics.md").read_text(encoding="utf-8")
    for figure in ("validation_alpha.png", "test_ndcg.png"):
        assert (output / "figures" / figure).stat().st_size > 0


def test_train_writes_model_and_metadata(data_dir, tmp_path):
    train = load_script("train")
    model_path = tmp_path / "models" / "recommender.joblib"
    assert train.main(["--data-dir", str(data_dir), "--output", str(model_path)]) == 0

    metadata = json.loads(model_path.with_name("metadata.json").read_text(encoding="utf-8"))
    ratings = pd.read_csv(data_dir / "ratings.csv")
    assert metadata["config"] == {
        "n_components": FINAL_CONFIG.n_components,
        "content_weighted": True,
        "alpha": FINAL_CONFIG.alpha,
        "c": None,
    }
    assert metadata["data"]["fingerprint"] == fingerprint(ratings)
    model = HybridRecommender.load(model_path)
    assert len(model.recommend_for_user(int(ratings["UserID"].iloc[0]))) == 10


def test_scripts_explain_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_script("train").main(["--data-dir", str(tmp_path), "--output", str(tmp_path / "m.joblib")])


def test_surprise_row_is_explained_when_the_package_is_missing(data_dir, tmp_path, monkeypatch, capsys):
    """Sem o scikit-surprise, a linha dele sai "não executado" e o usuário sabe por quê."""
    evaluate = load_script("evaluate")
    monkeypatch.setattr(evaluate, "surprise_available", lambda: False)
    output = tmp_path / "reports"
    evaluate.main(["--data-dir", str(data_dir), "--output-dir", str(output), "--no-figures"])
    assert "scikit-surprise não instalado" in capsys.readouterr().out
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["selection"]["surprise_params"] is None
    assert not (output / "figures").exists()


def test_check_model_accepts_the_notebook_model_and_rejects_others(data_dir, tmp_path, monkeypatch, capsys):
    """check_model.py compara o usuário 1 com o notebook; aqui, com dados sintéticos, a
    referência é trocada pela saída do próprio modelo para testar os dois caminhos."""
    train, check = load_script("train"), load_script("check_model")
    model_path = tmp_path / "recommender.joblib"
    train.main(["--data-dir", str(data_dir), "--output", str(model_path)])

    # Dados sintéticos: não é o modelo do notebook
    assert check.main(["--model", str(model_path)]) == 1
    assert "não batem com o notebook" in capsys.readouterr().err

    result = HybridRecommender.load(model_path).recommend_for_user(1, n=10)
    monkeypatch.setattr(check, "EXPECTED_MOVIES", result["MovieID"].tolist())
    monkeypatch.setattr(check, "EXPECTED_SCORES", result["Score"].tolist())
    assert check.main(["--model", str(model_path)]) == 0

    monkeypatch.setattr(check, "EXPECTED_SCORES", (result["Score"] + 0.01).tolist())
    assert check.main(["--model", str(model_path)]) == 1  # mesma lista, scores diferentes
    assert check.main(["--model", str(tmp_path / "nao-existe.joblib")]) == 1

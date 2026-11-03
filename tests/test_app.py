"""App Streamlit executado com o AppTest: abas, interações e estados vazios."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from recommender import HybridRecommender  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture(scope="module")
def model_file(tmp_path_factory, ratings, movies):
    path = tmp_path_factory.mktemp("app-model") / "recommender.joblib"
    HybridRecommender.fit(ratings, movies).save(path)
    return path


@pytest.fixture
def app(model_file, monkeypatch):
    monkeypatch.setenv("RECOMMENDER_MODEL_PATH", str(model_file))
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def test_opens_with_four_tabs_and_an_empty_state(app):
    assert [tab.label for tab in app.tabs] == [
        "Minhas recomendações",
        "Usuários do MovieLens",
        "Filmes parecidos",
        "Sobre o modelo",
    ]
    assert any("Escolha pelo menos um filme" in info.value for info in app.info)


def test_new_user_gets_recommendations(app, model_file):
    rec = HybridRecommender.load(model_file)
    chosen = rec.search_movies("", limit=3)["MovieID"].tolist()
    app.multiselect[0].set_value(chosen).run()
    for movie_id, rating in zip(chosen, [5, 4, 1], strict=True):
        app.slider(key=f"nota-{movie_id}").set_value(rating)
    app.button[0].click().run()

    assert not app.exception
    shown = app.dataframe[0].value
    assert len(shown) == 10
    expected = rec.recommend_for_ratings(dict(zip(chosen, [5, 4, 1], strict=True)), 10)
    assert shown["Afinidade"].round(6).tolist() == expected["Score"].round(6).tolist()
    assert any("70%" in caption.value for caption in app.caption)


def test_equal_ratings_are_explained(app, model_file):
    chosen = HybridRecommender.load(model_file).search_movies("", limit=2)["MovieID"].tolist()
    app.multiselect[0].set_value(chosen).run()
    app.button[0].click().run()  # notas padrão: todas iguais
    assert any("todas as suas notas são iguais" in caption.value for caption in app.caption)


def test_dataset_user_shows_history_and_recommendations(app):
    tables = app.tabs[1].dataframe
    assert len(tables) == 2
    assert "Sua nota" in tables[0].value.columns
    assert len(tables[1].value) == 10


def test_similar_movies(app, model_file):
    movie_id = HybridRecommender.load(model_file).search_movies("", limit=1)["MovieID"].iloc[0]
    app.selectbox[0].set_value(movie_id).run()
    assert not app.exception
    assert len(app.tabs[2].dataframe[0].value) == 10


def test_missing_model_explains_how_to_create_it(tmp_path, monkeypatch):
    monkeypatch.setenv("RECOMMENDER_MODEL_PATH", str(tmp_path / "nao-existe.joblib"))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "scripts/train.py" in at.error[0].value

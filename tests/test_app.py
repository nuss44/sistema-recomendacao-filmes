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


@pytest.fixture(scope="module")
def model(model_file):
    return HybridRecommender.load(model_file)


@pytest.fixture
def app(model_file, monkeypatch):
    monkeypatch.setenv("RECOMMENDER_MODEL_PATH", str(model_file))
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def stars(at, movie_id):
    return next(widget for widget in at.get("feedback") if widget.key == f"nota-{movie_id}")


def recommendations(at):
    return at.tabs[0].dataframe[0].value


def test_opens_with_four_tabs_and_an_empty_state(app):
    assert [tab.label for tab in app.tabs] == [
        "Minhas recomendações",
        "Usuários do MovieLens",
        "Filmes parecidos",
        "Sobre o modelo",
    ]
    assert any("Escolha alguns filmes" in info.value for info in app.info)
    assert app.button[0].label == "Usar um exemplo"
    assert not app.tabs[0].dataframe


def test_new_user_gets_recommendations_without_clicking(app, model):
    chosen = model.search_movies("", limit=3)["MovieID"].tolist()
    app.multiselect[0].set_value(chosen).run()
    for movie_id, rating in zip(chosen, [5, 4, 1], strict=True):
        stars(app, movie_id).set_value(rating - 1)  # as estrelas guardam 0 a 4
    app.run()

    assert not app.exception
    shown = recommendations(app)
    expected = model.recommend_for_ratings(dict(zip(chosen, [5, 4, 1], strict=True)), 10)
    assert len(shown) == 10
    assert shown["Afinidade"].round(6).tolist() == expected["Score"].round(6).tolist()
    assert any("70%" in caption.value for caption in app.caption)


def test_list_updates_when_a_rating_changes(app, model):
    chosen = model.search_movies("", limit=3)["MovieID"].tolist()
    app.multiselect[0].set_value(chosen).run()
    before = recommendations(app)["Filme"].tolist()

    stars(app, chosen[0]).set_value(0).run()  # 5 estrelas -> 1 estrela
    after = recommendations(app)["Filme"].tolist()
    assert len(after) == 10  # a lista continua na tela
    assert after != before


def test_example_button_fills_movies_and_ratings(app):
    app.button[0].click().run()
    assert not app.exception
    chosen = app.multiselect[0].value
    assert len(chosen) == 5
    ratings = [stars(app, movie_id).value + 1 for movie_id in chosen]
    assert ratings == [5, 5, 4, 2, 1]  # dados sintéticos: exemplo com os mais populares
    assert len(recommendations(app)) == 10


def test_list_size_can_change(app, model):
    app.multiselect[0].set_value(model.search_movies("", limit=2)["MovieID"].tolist()).run()
    app.selectbox(key="n-novo").set_value(20).run()
    assert len(recommendations(app)) == 20


def test_equal_ratings_are_explained(app, model):
    chosen = model.search_movies("", limit=2)["MovieID"].tolist()
    app.multiselect[0].set_value(chosen).run()  # notas iniciais: todas iguais
    assert any("todas as suas notas são iguais" in caption.value for caption in app.caption)


def test_dataset_user_shows_history_and_recommendations(app):
    tables = app.tabs[1].dataframe
    assert len(tables) == 2
    assert "Sua nota" in tables[0].value.columns
    assert len(tables[1].value) == 10


def test_similar_movies(app, model):
    movie_id = model.search_movies("", limit=1)["MovieID"].iloc[0]
    app.selectbox(key="similar").set_value(movie_id).run()
    assert not app.exception
    assert len(app.tabs[2].dataframe[0].value) == 10


def test_missing_model_explains_how_to_create_it(tmp_path, monkeypatch):
    monkeypatch.setenv("RECOMMENDER_MODEL_PATH", str(tmp_path / "nao-existe.joblib"))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "scripts/train.py" in at.error[0].value

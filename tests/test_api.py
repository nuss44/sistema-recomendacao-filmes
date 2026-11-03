"""API REST: respostas, validação de entrada e erros."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from recommender.api import create_app  # noqa: E402


@pytest.fixture(scope="module")
def model_path(tmp_path_factory, ratings, movies):
    from recommender import HybridRecommender

    path = tmp_path_factory.mktemp("model") / "recommender.joblib"
    HybridRecommender.fit(ratings, movies).save(path)
    path.with_name("metadata.json").write_text('{"config": {"alpha": 0.7}}', encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def client(model_path):
    with TestClient(create_app(model_path)) as client:
        yield client


@pytest.fixture(scope="module")
def known(movies, ratings):
    return {"movie": int(movies["MovieID"].iloc[0]), "user": int(ratings["UserID"].iloc[0])}


def test_health(client, movies):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["movies"] == len(movies)
    assert body["model"] == {"config": {"alpha": 0.7}}


def test_search_movies(client):
    response = client.get("/movies", params={"search": "FILME", "limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 5
    assert set(body[0]) == {"movie_id", "title", "genres", "year"}
    assert all("filme" in movie["title"].lower() for movie in body)


def test_get_movie(client, known):
    body = client.get(f"/movies/{known['movie']}").json()
    assert body["movie_id"] == known["movie"]
    assert isinstance(body["genres"], list) and body["year"] is not None


def test_similar_movies(client, known):
    body = client.get(f"/movies/{known['movie']}/similar", params={"n": 7}).json()
    assert len(body) == 7
    assert known["movie"] not in {m["movie_id"] for m in body}


def test_user_recommendations_match_the_model(client, known, model_path):
    from recommender import HybridRecommender

    body = client.get(f"/users/{known['user']}/recommendations").json()
    expected = HybridRecommender.load(model_path).recommend_for_user(known["user"])
    assert [m["movie_id"] for m in body] == expected["MovieID"].tolist()
    assert all(0 <= m["score"] <= 1 for m in body)


def test_new_user_recommendations(client, movies):
    rated = [int(m) for m in movies["MovieID"].iloc[:3]]
    payload = {"ratings": [{"movie_id": m, "rating": r} for m, r in zip(rated, [5, 4, 1], strict=True)], "n": 5}
    response = client.post("/recommendations", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 5
    assert not {m["movie_id"] for m in body} & set(rated)


@pytest.mark.parametrize(
    "request_args, status",
    [
        (("get", "/movies/999999"), 404),
        (("get", "/movies/999999/similar"), 404),
        (("get", "/users/999999/recommendations"), 404),
        (("get", "/users/1/recommendations?n=0"), 422),
        (("get", "/users/1/recommendations?n=51"), 422),
    ],
    ids=["filme-inexistente", "similar-inexistente", "usuario-inexistente", "n-zero", "n-grande"],
)
def test_errors(client, request_args, status):
    method, url = request_args
    assert getattr(client, method)(url).status_code == status


@pytest.mark.parametrize(
    "given, status",
    [
        ([], 422),  # nenhuma nota
        ([{"movie_id": "KNOWN", "rating": 6}], 422),  # nota fora de 1–5
        ([{"movie_id": "KNOWN", "rating": 5}, {"movie_id": "KNOWN", "rating": 3}], 422),  # filme repetido
        ([{"movie_id": 999999, "rating": 5}], 404),  # filme inexistente
    ],
    ids=["vazio", "nota-invalida", "repetido", "filme-inexistente"],
)
def test_new_user_input_is_validated(client, known, given, status):
    given = [{**r, "movie_id": known["movie"] if r["movie_id"] == "KNOWN" else r["movie_id"]} for r in given]
    response = client.post("/recommendations", json={"ratings": given})
    assert response.status_code == status
    if status == 404:
        assert "999999" in response.json()["detail"]


def test_without_model_the_api_explains_what_to_do(tmp_path):
    with TestClient(create_app(tmp_path / "nao-existe.joblib")) as client:
        assert client.get("/health").json()["status"] == "sem modelo"
        response = client.get("/movies")
        assert response.status_code == 503
        assert "train.py" in response.json()["detail"]

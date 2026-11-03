"""API REST do recomendador.

Para rodar (depois de `python scripts/train.py`):
    uvicorn recommender.api:app --reload
Documentação interativa: http://localhost:8000/docs

O modelo é carregado uma vez, na inicialização, e nunca é alterado pelas requisições:
notas de um usuário novo entram pelo POST /recommendations e são usadas no cálculo
(fold-in), sem gravar nada em disco.
"""

# Sem `from __future__ import annotations`: o FastAPI precisa das anotações como objetos
# para resolver as dependências (Model, N) definidas dentro de create_app.
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator

from . import __version__, catalog
from .config import MODEL_PATH
from .recommender import HybridRecommender


# ----------------------------------------------------------------------
# Esquemas
# ----------------------------------------------------------------------
class Movie(BaseModel):
    movie_id: int
    title: str = Field(description="Título legível, com o artigo no início")
    genres: list[str]
    year: int | None = None


class Recommendation(Movie):
    score: float = Field(description="Score do híbrido, de 0 a 1 (maior = mais recomendado)")


class RatedMovie(BaseModel):
    movie_id: int
    rating: float = Field(ge=1, le=5, description="Nota de 1 a 5")


class RecommendationRequest(BaseModel):
    ratings: list[RatedMovie] = Field(min_length=1, max_length=200)
    n: int = Field(default=10, ge=1, le=50)

    @field_validator("ratings")
    @classmethod
    def unique_movies(cls, ratings: list[RatedMovie]) -> list[RatedMovie]:
        ids = [r.movie_id for r in ratings]
        if len(ids) != len(set(ids)):
            raise ValueError("cada filme deve aparecer uma única vez")
        return ratings


class Health(BaseModel):
    status: str
    version: str
    movies: int | None = None
    users: int | None = None
    model: dict | None = Field(default=None, description="Metadados do treino (models/metadata.json)")


def _movies(frame: pd.DataFrame, with_score: bool = False) -> list:
    schema = Recommendation if with_score else Movie
    return [
        schema(
            movie_id=int(row.MovieID),
            title=catalog.display_title(row.Title),
            genres=row.Genres.split("|") if row.Genres else [],
            year=catalog.year_of(row.Title),
            **({"score": round(float(row.Score), 6)} if with_score else {}),
        )
        for row in frame.itertuples(index=False)
    ]


# ----------------------------------------------------------------------
# Aplicação
# ----------------------------------------------------------------------
def create_app(model_path: Path | str | None = None) -> FastAPI:
    """Cria a API. `model_path` permite usar outro modelo (testes, Docker)."""
    path = Path(model_path) if model_path is not None else MODEL_PATH

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = HybridRecommender.load(path) if path.exists() else None
        metadata = path.with_name("metadata.json")
        app.state.metadata = json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else None
        yield

    app = FastAPI(
        title="Recomendador de Filmes",
        description="Recomendador híbrido (SVD + conteúdo) treinado no MovieLens 1M.",
        version=__version__,
        lifespan=lifespan,
    )

    def model(request: Request) -> HybridRecommender:
        if request.app.state.model is None:
            raise HTTPException(503, f"Modelo não encontrado em {path}. Rode `python scripts/train.py`.")
        return request.app.state.model

    Model = Annotated[HybridRecommender, Depends(model)]
    N = Annotated[int, Query(ge=1, le=50, description="Tamanho da lista")]

    @app.get("/health", response_model=Health, tags=["status"])
    def health(request: Request) -> Health:
        rec = request.app.state.model
        if rec is None:
            return Health(status="sem modelo", version=__version__)
        return Health(
            status="ok",
            version=__version__,
            movies=rec.index.n_items,
            users=rec.index.n_users,
            model=request.app.state.metadata,
        )

    @app.get("/movies", response_model=list[Movie], tags=["filmes"])
    def search_movies(
        rec: Model,
        search: Annotated[str, Query(max_length=100, description="Trecho do título")] = "",
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ) -> list[Movie]:
        """Busca por título, sem acento e sem diferenciar maiúsculas. Vazio = mais populares."""
        return _movies(rec.search_movies(search, limit))

    @app.get("/movies/{movie_id}", response_model=Movie, tags=["filmes"])
    def get_movie(movie_id: int, rec: Model) -> Movie:
        try:
            return _movies(rec.movie(movie_id).to_frame().T)[0]
        except KeyError as error:
            raise HTTPException(404, f"Filme {movie_id} não encontrado") from error

    @app.get("/movies/{movie_id}/similar", response_model=list[Recommendation], tags=["filmes"])
    def similar_movies(movie_id: int, rec: Model, n: N = 10) -> list[Recommendation]:
        """Filmes com gêneros mais parecidos; empates desfeitos pela popularidade."""
        try:
            return _movies(rec.similar_movies(movie_id, n), with_score=True)
        except KeyError as error:
            raise HTTPException(404, f"Filme {movie_id} não encontrado") from error

    @app.get("/users/{user_id}/recommendations", response_model=list[Recommendation], tags=["recomendações"])
    def user_recommendations(user_id: int, rec: Model, n: N = 10) -> list[Recommendation]:
        """Recomendações para um usuário do MovieLens (sem os filmes que ele já avaliou)."""
        try:
            return _movies(rec.recommend_for_user(user_id, n), with_score=True)
        except KeyError as error:
            raise HTTPException(404, f"Usuário {user_id} não encontrado") from error

    @app.post("/recommendations", response_model=list[Recommendation], tags=["recomendações"])
    def recommendations(body: RecommendationRequest, rec: Model) -> list[Recommendation]:
        """Recomendações para um usuário novo, a partir das notas que ele deu."""
        ratings = {r.movie_id: r.rating for r in body.ratings}
        try:
            return _movies(rec.recommend_for_ratings(ratings, body.n), with_score=True)
        except KeyError as error:
            raise HTTPException(404, str(error).strip("'\"")) from error

    return app


app = create_app()

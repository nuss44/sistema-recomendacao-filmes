from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Inicializar API
app = FastAPI(title="Movie Recommender API")

# Paths
DATA_PATH = "./data"
MODEL_PATH = "./models"

# Carregar dados
movies = pd.read_csv(os.path.join(DATA_PATH, "movies.csv"))
ratings = pd.read_csv(os.path.join(DATA_PATH, "ratings.csv"))
users = pd.read_csv(os.path.join(DATA_PATH, "users.csv"))

# Carregar modelo treinado
model = joblib.load(os.path.join(MODEL_PATH, "model.pkl"))

# Schemas
class RecommendationRequest(BaseModel):
    user_id: int
    n_recommendations: int = 5

class User(BaseModel):
    user_id: int
    name: str = "Anonymous"

class Rating(BaseModel):
    user_id: int
    movie_id: int
    rating: float

# Rotas
@app.get("/")
def home():
    return {"message": "API de Recomendação de Filmes funcionando"}

# Recomendação por usuário
@app.get("/recommend/user/{user_id}")
def recommend_movies(user_id: int, n: int = 5):
    if user_id not in users['UserID'].values:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")

    user_rated_movies = ratings[ratings['UserID'] == user_id]['MovieID'].tolist()
    candidate_movies = movies[~movies['MovieID'].isin(user_rated_movies)].copy()

    candidate_movies['predicted_rating'] = candidate_movies['MovieID'].apply(
        lambda x: model.predict(user_id, x).est
    )

    top_movies = candidate_movies.sort_values(by="predicted_rating", ascending=False).head(n)
    recommendations = top_movies[['MovieID', 'Title', 'predicted_rating']].to_dict(orient='records')

    return {
        "user_id": user_id,
        "n_recommendations": n,
        "recommendations": recommendations
    }

# Itens similares a um filme
@app.get("/recommend/item/{item_id}")
def recommend_similar_items(item_id: int, n: int = 5):
    if item_id not in movies['MovieID'].values:
        raise HTTPException(status_code=404, detail="Filme não encontrado")

    target_genres = movies[movies['MovieID'] == item_id]['Genres'].values[0]
    similar_movies = movies[movies['Genres'] == target_genres]
    similar_movies = similar_movies[similar_movies['MovieID'] != item_id]

    top_similar = similar_movies.head(n)
    results = top_similar[['MovieID', 'Title', 'Genres']].to_dict(orient='records')

    return {
        "item_id": item_id,
        "n_similar": n,
        "similar_items": results
    }

# Adicionar usuário
@app.post("/user/add")
def add_user(user: User):
    if user.user_id in users['UserID'].values:
        raise HTTPException(status_code=400, detail="Usuário já existe")

    new_user = {'UserID': user.user_id, 'Name': user.name}
    users.loc[len(users)] = new_user
    users.to_csv(os.path.join(DATA_PATH, "users.csv"), index=False)

    return {
        "message": "Usuário adicionado com sucesso",
        "user": new_user
    }

# Atualizar preferências (nota)
@app.post("/user/update_preferences")
def update_preferences(rating: Rating):
    if rating.user_id not in users['UserID'].values:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")
    if rating.movie_id not in movies['MovieID'].values:
        raise HTTPException(status_code=404, detail="Filme não encontrado")

    new_rating = {
        'UserID': rating.user_id,
        'MovieID': rating.movie_id,
        'Rating': rating.rating
    }

    ratings.loc[len(ratings)] = new_rating
    ratings.to_csv(os.path.join(DATA_PATH, "ratings.csv"), index=False)

    return {
        "message": "Preferência atualizada com sucesso",
        "rating": new_rating
    }

# Listar todos os filmes
@app.get("/movies/")
def get_movies():
    return movies.to_dict(orient="records")

# Listar todos os usuários
@app.get("/users/")
def get_users():
    return users.to_dict(orient="records")

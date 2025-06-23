import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV
import joblib

# Carregar dados
ratings = pd.read_csv('data/ratings.csv')

# Preparar os dados
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['UserID', 'MovieID', 'Rating']], reader)

# Definir grid de hiperparâmetros
param_grid = {
    'n_factors': [50, 100, 150],
    'lr_all': [0.002, 0.005, 0.01],
    'reg_all': [0.02, 0.05, 0.1],
    'n_epochs': [20, 50, 70]
}

# Grid Search
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
gs.fit(data)

# Resultado
print(f"Melhores parâmetros: {gs.best_params['rmse']}")
print(f"Melhor RMSE: {gs.best_score['rmse']}")

# Treinar modelo final
best_model = gs.best_estimator['rmse']
best_model.fit(data.build_full_trainset())
# Salvar modelo
joblib.dump(best_model, 'models.pkl')   
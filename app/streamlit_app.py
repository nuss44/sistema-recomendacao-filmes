"""Interface do recomendador de filmes.

Para rodar (depois de `python scripts/train.py`):
    streamlit run app/streamlit_app.py

O app usa o pacote `recommender` diretamente, sem depender da API.
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from recommender import HybridRecommender, catalog
from recommender.config import MODEL_PATH, REPORTS_DIR

st.set_page_config(page_title="Recomendador de filmes", page_icon="🎬", layout="wide")


# ----------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando o modelo…")
def load_model(path: str) -> HybridRecommender:
    return HybridRecommender.load(path)


def model_path() -> Path:
    # Lido a cada execução (e não só na importação) para permitir trocar o modelo nos testes e no Docker
    return Path(os.environ.get("RECOMMENDER_MODEL_PATH", MODEL_PATH))


def as_table(frame: pd.DataFrame, score: bool = True) -> pd.DataFrame:
    """Formato de exibição: título legível, ano, gêneros e afinidade."""
    table = pd.DataFrame(
        {
            "Filme": frame["Title"].map(catalog.display_title),
            "Ano": frame["Title"].map(catalog.year_of),
            "Gêneros": frame["Genres"].str.replace("|", ", ", regex=False),
        }
    )
    if score:
        table["Afinidade"] = frame["Score"].to_numpy()
    if "Rating" in frame:
        table["Sua nota"] = frame["Rating"].map(lambda r: "★" * int(r))
    return table


def show_list(frame: pd.DataFrame, score: bool = True) -> None:
    st.dataframe(
        as_table(frame, score),
        hide_index=True,
        width="stretch",
        column_config={
            "Ano": st.column_config.NumberColumn(format="%d", width="small"),
            "Afinidade": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="%.2f", help="Score do modelo, de 0 a 1"
            ),
        },
    )


def explain(rec: HybridRecommender, ratings: dict[int, float] | None = None) -> str:
    config = rec.config
    if config.alpha is None:
        return "O peso entre as duas partes depende de quantos filmes você avaliou."
    if ratings is not None and len(set(ratings.values())) == 1:
        return (
            "Como todas as suas notas são iguais, a lista vem só do padrão de usuários parecidos com você. "
            "Dê notas diferentes para os gêneros também contarem."
        )
    collaborative = round(config.alpha * 100)
    return (
        f"Como a lista é calculada: {collaborative}% vem de usuários com gosto parecido com o seu e "
        f"{100 - collaborative}% dos gêneros dos filmes que você avaliou bem."
    )


# ----------------------------------------------------------------------
# Página
# ----------------------------------------------------------------------
st.title("Recomendador de filmes")
st.write(
    "Avalie filmes que você já viu e receba sugestões calculadas por um modelo treinado com "
    "1 milhão de avaliações do MovieLens (filmes lançados até 2000)."
)

path = model_path()
if not path.exists():
    st.error(
        f"Modelo não encontrado em `{path}`. Gere-o com:\n\n"
        "```\npython scripts/download_data.py\npython scripts/train.py\n```"
    )
    st.stop()

rec = load_model(str(path))
by_popularity = rec.search_movies("", limit=rec.index.n_items)
titles = dict(zip(by_popularity["MovieID"], by_popularity["Title"].map(catalog.display_title), strict=True))

mine, dataset, similar, about = st.tabs(
    ["Minhas recomendações", "Usuários do MovieLens", "Filmes parecidos", "Sobre o modelo"]
)

# ---------------- Usuário novo ----------------
with mine:
    chosen = st.multiselect(
        "Filmes que você já viu",
        options=list(titles),
        format_func=titles.get,
        placeholder="Digite parte do título, em inglês",
        help="Os filmes mais populares aparecem primeiro.",
    )
    if not chosen:
        st.info("Escolha pelo menos um filme para começar. Filmes de que você não gostou também ajudam.")
    else:
        st.write("Dê uma nota de 1 a 5 para cada filme:")
        ratings = {}
        columns = st.columns(min(len(chosen), 3))
        for i, movie_id in enumerate(chosen):
            with columns[i % len(columns)]:
                ratings[movie_id] = st.slider(titles[movie_id], 1, 5, 4, key=f"nota-{movie_id}")
        n = st.slider("Quantos filmes recomendar", 5, 30, 10, key="n-novo")
        if st.button("Recomendar filmes", type="primary"):
            st.caption(explain(rec, ratings))
            show_list(rec.recommend_for_ratings(ratings, n))

# ---------------- Usuário do dataset ----------------
with dataset:
    st.write("Veja o que o modelo recomenda para quem está no MovieLens, ao lado do que essa pessoa avaliou.")
    user_id = st.number_input(
        "Número do usuário",
        min_value=int(rec.user_ids.min()),
        max_value=int(rec.user_ids.max()),
        value=int(rec.user_ids.min()),
        step=1,
    )
    if user_id not in set(rec.user_ids.tolist()):
        st.warning(f"O usuário {user_id} não está no conjunto de treino.")
    else:
        left, right = st.columns(2)
        with left:
            history = rec.user_history(int(user_id))
            st.subheader("Avaliou com as maiores notas")
            st.caption(f"{len(history)} filmes avaliados no total.")
            show_list(history.head(10), score=False)
        with right:
            st.subheader("Recomendações")
            st.caption(explain(rec))
            show_list(rec.recommend_for_user(int(user_id), 10))

# ---------------- Filmes parecidos ----------------
with similar:
    st.write("Filmes com os gêneros mais parecidos. Entre os empatados, os mais populares vêm primeiro.")
    movie_id = st.selectbox(
        "Filme",
        options=list(titles),
        format_func=titles.get,
        index=None,
        placeholder="Digite parte do título, em inglês",
    )
    if movie_id is None:
        st.info("Escolha um filme para ver os parecidos.")
    else:
        st.caption(f"Gêneros: {rec.movie(movie_id)['Genres'].replace('|', ', ')}")
        show_list(rec.similar_movies(movie_id, 10))

# ---------------- Sobre ----------------
with about:
    config = rec.config
    st.subheader("Como funciona")
    st.write(
        f"Um modelo híbrido combina duas fontes. A primeira é uma fatoração de matrizes (SVD truncado com "
        f"{config.n_components} componentes), que encontra usuários com gosto parecido. A segunda compara os "
        "gêneros dos filmes, ponderados pelas notas que a pessoa deu. Os dois scores são normalizados de 0 a 1 "
        "e somados com pesos escolhidos na validação."
    )
    metrics_file = REPORTS_DIR / "metrics.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        test = pd.DataFrame(metrics["test"]).set_index("Modelo")
        st.subheader("Desempenho no teste")
        st.write(
            "Avaliação top-10 nas avaliações mais recentes de cada usuário, que o modelo nunca viu. "
            "Uma recomendação conta como acerto quando o usuário deu nota 4 ou 5 ao filme."
        )
        st.dataframe(test.dropna(), width="stretch", column_config={"Modelo": "Modelo"})
        st.caption(f"Fonte: `reports/metrics.json`, gerado em {metrics['generated_at']}.")
        figure = REPORTS_DIR / "figures" / "test_ndcg.png"
        if figure.exists():
            st.image(str(figure))
    else:
        st.info("Rode `python scripts/evaluate.py` para gerar as métricas.")

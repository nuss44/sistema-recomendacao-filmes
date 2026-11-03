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
    """Formato de exibição: título legível (já inclui o ano), gêneros e afinidade."""
    table = pd.DataFrame(
        {
            "Filme": frame["Title"].map(catalog.display_title),
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
            "Filme": st.column_config.TextColumn(width="medium"),
            "Gêneros": st.column_config.TextColumn(width="medium"),
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
st.title("Recomendador de filmes", anchor=False)
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
# Exemplo para quem abre o app pela primeira vez: gosta de animação e ficção científica,
# não gosta de romance. MovieIDs do MovieLens 1M; se faltarem, usa os mais populares.
EXAMPLE = {1: 5, 2355: 4, 260: 5, 2571: 4, 1721: 1}


def use_example() -> None:
    example = {m: r for m, r in EXAMPLE.items() if m in titles}
    if len(example) < 3:
        example = dict(zip(list(titles)[:5], [5, 5, 4, 2, 1], strict=False))
    st.session_state["escolhidos"] = list(example)
    for movie_id, rating in example.items():
        st.session_state[f"nota-{movie_id}"] = rating - 1  # as estrelas guardam 0 a 4


with mine:
    chosen = st.multiselect(
        "Filmes que você já viu",
        options=list(titles),
        format_func=titles.get,
        key="escolhidos",
        placeholder="Digite parte do título, em inglês",
        help="Os filmes mais populares aparecem primeiro.",
    )
    if not chosen:
        st.info("Escolha alguns filmes e dê uma nota a cada um. Filmes de que você não gostou também ajudam.")
        st.button("Usar um exemplo", on_click=use_example)
    else:
        left, right = st.columns([5, 7], gap="large")
        with left:
            st.subheader("Suas notas", anchor=False)
            ratings = {}
            for movie_id in chosen:
                title_column, stars_column = st.columns([3, 2], vertical_alignment="center")
                title_column.write(titles[movie_id])
                with stars_column:
                    # Valor inicial pelo session_state (não por `default`), pois o exemplo também o define
                    st.session_state.setdefault(f"nota-{movie_id}", 3)  # 4 estrelas
                    stars = st.feedback("stars", key=f"nota-{movie_id}")
                if stars is not None:
                    ratings[movie_id] = stars + 1
        with right:
            header, size = st.columns([3, 1], vertical_alignment="bottom")
            header.subheader("Recomendações", anchor=False)
            n = size.selectbox(
                "Quantos", [10, 20, 30], key="n-novo", format_func="{} filmes".format, label_visibility="collapsed"
            )
            if not ratings:
                st.info("Dê uma nota a pelo menos um filme.")
            else:
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
        width=200,
    )
    if user_id not in set(rec.user_ids.tolist()):
        st.warning(f"O usuário {user_id} não está no conjunto de treino.")
    else:
        left, right = st.columns(2)
        with left:
            history = rec.user_history(int(user_id))
            st.subheader("Avaliou com as maiores notas", anchor=False)
            st.caption(f"{len(history)} filmes avaliados no total.")
            show_list(history.head(10), score=False)
        with right:
            st.subheader("Recomendações", anchor=False)
            st.caption(explain(rec))
            show_list(rec.recommend_for_user(int(user_id), 10))

# ---------------- Filmes parecidos ----------------
with similar:
    st.write("Filmes com os gêneros mais parecidos. Entre os empatados, os mais populares vêm primeiro.")
    movie_id = st.selectbox(
        "Filme",
        key="similar",
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
    st.subheader("Como funciona", anchor=False)
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
        st.subheader("Desempenho no teste", anchor=False)
        st.write(
            "Avaliação top-10 nas avaliações mais recentes de cada usuário, que o modelo nunca viu. "
            "Uma recomendação conta como acerto quando o usuário deu nota 4 ou 5 ao filme."
        )
        test = test.dropna().assign(Cobertura=lambda t: t["Cobertura"] * 100)
        decimals = st.column_config.NumberColumn(format="%.4f")
        st.dataframe(
            test,
            width="stretch",
            column_config={
                "Precision@10": decimals,
                "Recall@10": decimals,
                "NDCG@10": decimals,
                "Cobertura": st.column_config.NumberColumn(format="%.1f%%", help="% do catálogo recomendado"),
            },
        )
        generated = pd.Timestamp(metrics["generated_at"]).tz_convert("America/Sao_Paulo")
        st.caption(f"Fonte: `reports/metrics.json`, gerado em {generated:%d/%m/%Y às %H:%M}.")
        figure = REPORTS_DIR / "figures" / "test_ndcg.png"
        if figure.exists():
            st.image(str(figure))
    else:
        st.info("Rode `python scripts/evaluate.py` para gerar as métricas.")

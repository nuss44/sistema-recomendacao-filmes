# Imagem com o app (Streamlit), a API (FastAPI) e o modelo já treinado.
#
#   docker compose up --build
#   app: http://localhost:8501 · API: http://localhost:8000/docs
#
# O build baixa o MovieLens 1M do GroupLens, treina o modelo e confere que ele é idêntico
# ao do notebook (scripts/check_model.py): se não for, o build falha. Os dados brutos não
# ficam na imagem final, só o modelo, que ainda contém os títulos e as notas de treino.
# Por isso a imagem é para uso local: a licença do MovieLens não permite redistribuí-la.

ARG PYTHON_VERSION=3.12

# ---------- Base: dependências fixadas e o pacote ----------
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
# Dependências antes do código: esta camada só é refeita quando requirements.txt muda.
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# Instalação editável: os caminhos do projeto (models/, reports/) ficam relativos a /app.
RUN python -m pip install --no-deps -e .

# ---------- Treino: baixa os dados, treina e confere ----------
FROM base AS model
ARG MOVIELENS_URL=https://files.grouplens.org/datasets/movielens/ml-1m.zip
# Use --build-arg VERIFY_MODEL=false só para treinar com outros dados.
ARG VERIFY_MODEL=true
COPY scripts ./scripts
RUN python scripts/download_data.py --url "$MOVIELENS_URL" \
 && python scripts/train.py \
 && if [ "$VERIFY_MODEL" = "true" ]; then python scripts/check_model.py; fi

# ---------- Execução: app e API ----------
FROM base AS runtime
COPY .streamlit ./.streamlit
COPY app ./app
COPY reports ./reports
COPY --from=model /app/models ./models
RUN useradd --create-home appuser
USER appuser
EXPOSE 8501 8000
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]

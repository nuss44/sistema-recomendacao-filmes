# Sistema de Recomendação de Filmes

[![CI](https://github.com/nuss44/sistema-recomendacao-filmes/actions/workflows/ci.yml/badge.svg)](https://github.com/nuss44/sistema-recomendacao-filmes/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)
[![Licença: MIT](https://img.shields.io/badge/licen%C3%A7a-MIT-green)](LICENSE)

Recomendador híbrido de filmes (fatoração de matrizes + conteúdo) treinado no MovieLens 1M,
com avaliação top-N reprodutível: split cronológico, escolhas só na validação e uma única
avaliação no teste.

> 🚧 Documentação completa em construção. As métricas oficiais estão em
> [`reports/metrics.md`](reports/metrics.md), geradas por código a partir dos dados.

## Como rodar

### Com Docker (mais simples)

```bash
docker compose up --build
```

O primeiro build baixa o MovieLens 1M, treina o modelo e confere que ele é idêntico ao do
notebook. Depois: app em http://localhost:8501 e API em http://localhost:8000/docs.

### Com Python (3.10 ou mais recente)

```bash
python -m venv .venv
source .venv/bin/activate              # Windows (Git Bash): source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,surprise,api,app]" -c requirements-dev.txt

python scripts/download_data.py        # baixa e valida o MovieLens 1M
python scripts/evaluate.py             # regenera reports/ (métricas e gráficos)
python scripts/train.py                # treina o modelo final em models/
python scripts/check_model.py          # confere que o modelo é igual ao do notebook
python -m pytest                       # testes

streamlit run app/streamlit_app.py     # interface: http://localhost:8501
uvicorn recommender.api:app            # API: http://localhost:8000/docs
```

`requirements.txt` (execução) e `requirements-dev.txt` (desenvolvimento) travam as versões
testadas. Para atualizá-las: `pip install uv` e rode o comando do cabeçalho de cada arquivo.

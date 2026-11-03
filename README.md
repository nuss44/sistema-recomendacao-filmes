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

```bash
python -m venv .venv
source .venv/bin/activate              # Windows (Git Bash): source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,surprise,api,app]"

python scripts/download_data.py        # baixa e valida o MovieLens 1M
python scripts/evaluate.py             # regenera reports/ (métricas e gráficos)
python scripts/train.py                # treina o modelo final em models/
python -m pytest                       # testes

streamlit run app/streamlit_app.py     # interface: http://localhost:8501
uvicorn recommender.api:app            # API: http://localhost:8000/docs
```

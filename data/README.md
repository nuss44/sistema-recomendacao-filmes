# Dados

Os dados **não são versionados**: a licença do MovieLens 1M não permite redistribuí-los.
Para obtê-los, rode a partir da raiz do projeto:

```bash
python scripts/download_data.py
```

O script baixa o arquivo oficial do GroupLens, converte para CSV e confere as contagens.

## Estrutura gerada

```
data/
├── raw/
│   ├── ml-1m.zip          # arquivo original do GroupLens
│   └── ml-1m/*.dat        # arquivos extraídos (separador "::", codificação latin-1)
└── processed/
    ├── movies.csv         # MovieID, Title, Genres                     (3.883 linhas)
    ├── users.csv          # UserID, Gender, Age, Occupation, Zip-code  (6.040 linhas)
    └── ratings.csv        # UserID, MovieID, Rating, Timestamp         (1.000.209 linhas)
```

## Fonte e licença

F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*.
ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19.
https://doi.org/10.1145/2827872

Termos de uso: https://files.grouplens.org/datasets/movielens/ml-1m-README.txt

# Métricas

> Arquivo gerado por `python scripts/evaluate.py`. Não edite à mão.

- Configuração escolhida na validação: SVD com 20 componentes, conteúdo ponderado por nota, α = 0.7
- Híbrido final no teste: NDCG@10 = 0,1213, cobertura = 21,9% (popularidade: 0,0907, 3,0%)
- Dados: 1.000.209 avaliações, impressão digital `fc8abd2acf4ff373`

## Teste (avaliação única, modelos re-treinados em treino + validação)

| Modelo | Precision@10 | Recall@10 | NDCG@10 | Cobertura |
|---|---|---|---|---|
| SVD sklearn (20 comp.) | 0,0992 | 0,0847 | 0,1218 | 0,1774 |
| Híbrido final (config. da validação) | 0,0997 | 0,0844 | 0,1213 | 0,2189 |
| Popularidade | 0,0799 | 0,0473 | 0,0907 | 0,0301 |
| Híbrido inicial (α=0.5, sem normalizar) | 0,0381 | 0,0400 | 0,0520 | 0,4569 |
| Conteúdo (ponderado por nota) | 0,0122 | 0,0094 | 0,0143 | 0,9825 |
| Conteúdo (original) | 0,0106 | 0,0106 | 0,0128 | 0,8514 |
| Aleatório | 0,0050 | 0,0028 | 0,0054 | 1,0000 |
| SVD Surprise (nota prevista) | não executado | não executado | não executado | não executado |

## Validação (usada para as escolhas)

| Modelo | Precision@10 | Recall@10 | NDCG@10 | Cobertura |
|---|---|---|---|---|
| Aleatório | 0,0038 | 0,0025 | 0,0042 | 1,0000 |
| Popularidade | 0,0676 | 0,0520 | 0,0796 | 0,0252 |
| SVD sklearn (20 comp.) | 0,0801 | 0,0882 | 0,1027 | 0,1617 |
| SVD sklearn (50 comp.) | 0,0761 | 0,0905 | 0,0984 | 0,2163 |
| SVD sklearn (100 comp.) | 0,0714 | 0,0891 | 0,0940 | 0,2640 |
| SVD sklearn (150 comp.) | 0,0664 | 0,0835 | 0,0883 | 0,2933 |
| SVD sklearn (200 comp.) | 0,0615 | 0,0783 | 0,0838 | 0,3235 |
| Conteúdo (original) | 0,0110 | 0,0119 | 0,0143 | 0,8563 |
| Conteúdo (ponderado por nota) | 0,0107 | 0,0084 | 0,0124 | 0,9791 |

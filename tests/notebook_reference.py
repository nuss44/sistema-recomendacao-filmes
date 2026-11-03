"""Código do notebook 01_modelagem_e_avaliacao.ipynb, copiado sem alterações.

Serve de referência para os testes de equivalência: o pacote `recommender` precisa
produzir exatamente os mesmos resultados que estas funções.

Cada bloco é o texto da célula indicada, sem alterações na lógica. As únicas
mudanças foram retirar prints/demonstrações e tornar o import do Surprise opcional.
Se o notebook mudar, atualize a célula correspondente aqui.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd

CELLS: dict[str, str] = {}

# Célula 10 — split cronológico
CELLS["split cronológico"] = (
    "import numpy as np\nimport pandas as pd\nfrom scipy.sparse import csr_matrix\n\n# ------------------------------------------------------------\n# Split cronológico por usuário: treino | validação | teste\n# ------------------------------------------------------------\nTEST_FRAC = 0.2   # 20% mais recentes de cada usuário -> teste\nVAL_FRAC = 0.2    # 20% mais recentes do que sobrou -> validação\n\ndef separar_ultimos(df, frac):\n    \"\"\"Separa as `frac` avaliações mais recentes de cada usuário (mínimo 1).\"\"\"\n    df = df.sort_values(['UserID', 'Timestamp'])\n    pos_do_fim = df.groupby('UserID').cumcount(ascending=False)\n    n = df.groupby('UserID')['MovieID'].transform('size')\n    n_corte = np.maximum(1, np.round(n * frac)).astype(int)\n    return df[pos_do_fim >= n_corte], df[pos_do_fim < n_corte]\n\ndev_df, test_df = separar_ultimos(ratings, TEST_FRAC)\ntrain_df, val_df = separar_ultimos(dev_df, VAL_FRAC)\n\n# Índices fixos de usuários e filmes (filmes na mesma ordem de `movies`)\nuser_ids = np.sort(ratings['UserID'].unique())\nitem_ids = movies['MovieID'].values\nu_idx = pd.Series(np.arange(len(user_ids)), index=user_ids)\ni_idx = pd.Series(np.arange(len(item_ids)), index=item_ids)\nn_users, n_items = len(user_ids), len(item_ids)\n\ndef matriz(df):\n    \"\"\"Matriz esparsa usuário x filme com as notas de `df`.\"\"\"\n    return csr_matrix(\n        (df['Rating'].astype(np.float32).values,\n         (u_idx[df['UserID']].values, i_idx[df['MovieID']].values)),\n        shape=(n_users, n_items))"
)

# Célula 25 — TF-IDF e similaridade
CELLS["TF-IDF e similaridade"] = (
    "from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.metrics.pairwise import cosine_similarity\n\n# Preparar os dados de gêneros\nmovies['Genres'] = movies['Genres'].fillna('')\n\n# Vetorização dos gêneros usando TF-IDF\ntfidf = TfidfVectorizer(token_pattern=r'[^|]+')\ntfidf_matrix = tfidf.fit_transform(movies['Genres'])\n\n# Calcular a similaridade de cosseno entre filmes\ncosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)"
)

# Célula 43 — avaliação
CELLS["avaliação"] = (
    '# ------------------------------------------------------------\n# Funções de avaliação (servem tanto para a validação quanto para o teste)\n# ------------------------------------------------------------\nK = 10\nLIMIAR_RELEVANTE = 4\nDESCONTO = 1 / np.log2(np.arange(2, K + 2))   # pesos de posição do NDCG\n\ndef relevantes_de(df):\n    """Para cada usuário (índice de linha), os filmes (índices de coluna) com nota >= limiar."""\n    rel = df[df[\'Rating\'] >= LIMIAR_RELEVANTE]\n    return (rel.assign(u=u_idx[rel[\'UserID\']].values, i=i_idx[rel[\'MovieID\']].values)\n               .groupby(\'u\')[\'i\'].apply(np.array))\n\ndef vistos_de(R):\n    """Máscara densa (usuários x filmes) do que já foi avaliado na matriz R."""\n    return R.toarray() > 0\n\ndef top_k(scores, vistos, k=K):\n    """Índices dos k filmes de maior score por usuário, ignorando os já vistos."""\n    s = np.array(scores, dtype=np.float32, copy=True)\n    s[vistos] = -np.inf\n    s[np.isnan(s)] = -np.inf\n    top = np.argpartition(-s, k, axis=1)[:, :k]\n    linhas = np.arange(s.shape[0])[:, None]\n    return top[linhas, np.argsort(-s[linhas, top], axis=1)]\n\ndef avaliar(scores, nome, vistos, relevantes):\n    """Precision@K, Recall@K, NDCG@K e cobertura de catálogo."""\n    top = top_k(scores, vistos)\n    precs, recs, ndcgs = [], [], []\n    for u, rel in relevantes.items():\n        acertos = np.isin(top[u], rel)\n        precs.append(acertos.sum() / K)\n        recs.append(acertos.sum() / len(rel))\n        ndcgs.append((acertos * DESCONTO).sum() / DESCONTO[:min(len(rel), K)].sum())\n    return {\'Modelo\': nome,\n            f\'Precision@{K}\': np.mean(precs),\n            f\'Recall@{K}\': np.mean(recs),\n            f\'NDCG@{K}\': np.mean(ndcgs),\n            \'Cobertura\': len(np.unique(top[relevantes.index.values])) / n_items}'
)

# Célula 45 — modelos
CELLS["modelos"] = (
    '# ------------------------------------------------------------\n# Modelos como funções: recebem os dados de ajuste e devolvem uma\n# matriz de scores (usuários x filmes). As MESMAS funções são usadas\n# na validação, no teste e no recomendador final.\n# ------------------------------------------------------------\nfrom sklearn.decomposition import TruncatedSVD\ntry:\n    from surprise import Dataset, Reader, SVD\nexcept ImportError:  # Surprise é opcional\n    pass\n\ndef score_aleatorio(R, seed=42):\n    return np.random.default_rng(seed).random(R.shape, dtype=np.float32)\n\ndef score_popularidade(R):\n    """Quantos usuários avaliaram cada filme (igual para todos os usuários)."""\n    pop = np.asarray((R > 0).sum(axis=0)).ravel().astype(np.float32)\n    return np.tile(pop, (R.shape[0], 1))\n\ndef score_svd_sklearn(R, n_components):\n    """TruncatedSVD sobre a matriz de notas (0 = não avaliado), como no notebook."""\n    modelo = TruncatedSVD(n_components=n_components, random_state=42)\n    fatores = modelo.fit_transform(R)\n    return (fatores @ modelo.components_).astype(np.float32)\n\ndef score_surprise(df, params):\n    """Nota prevista pelo SVD do Surprise para todos os pares usuário x filme."""\n    ts = Dataset.load_from_df(df[[\'UserID\', \'MovieID\', \'Rating\']],\n                              Reader(rating_scale=(1, 5))).build_full_trainset()\n    m = SVD(**params, random_state=42)\n    m.fit(ts)\n    # Montamos a matriz direto dos fatores: média + viés usuário + viés filme + p_u · q_i\n    P = np.zeros((n_users, params[\'n_factors\']), np.float32); bu = np.zeros(n_users, np.float32)\n    Q = np.zeros((n_items, params[\'n_factors\']), np.float32); bi = np.zeros(n_items, np.float32)\n    for raw_uid in df[\'UserID\'].unique():\n        inner = ts.to_inner_uid(raw_uid)\n        P[u_idx[raw_uid]], bu[u_idx[raw_uid]] = m.pu[inner], m.bu[inner]\n    for inner in ts.all_items():\n        j = i_idx[ts.to_raw_iid(inner)]\n        Q[j], bi[j] = m.qi[inner], m.bi[inner]\n    return ts.global_mean + bu[:, None] + bi[None, :] + P @ Q.T\n\ndef score_conteudo(R, ponderado):\n    """Perfil do usuário = soma das similaridades (gêneros) com os filmes que ele viu.\n    ponderado=True: cada filme pesa (nota - média do usuário), então filmes de que\n    ele não gostou passam a afastar recomendações parecidas."""\n    if not ponderado:\n        pesos = (R > 0).astype(np.float32)\n    else:\n        coo = R.tocoo()\n        media = np.asarray(R.sum(axis=1)).ravel() / np.maximum(1, R.getnnz(axis=1))\n        pesos = csr_matrix((coo.data - media[coo.row], (coo.row, coo.col)), shape=R.shape)\n    return np.asarray(pesos @ cosine_sim, dtype=np.float32)\n\ndef minmax_por_usuario(scores, vistos):\n    """Normaliza os scores de cada usuário para [0, 1], considerando só os filmes não vistos."""\n    s = np.array(scores, dtype=np.float32, copy=True)\n    s[vistos] = np.nan\n    mn = np.nanmin(s, axis=1, keepdims=True)\n    mx = np.nanmax(s, axis=1, keepdims=True)\n    return (s - mn) / (mx - mn + 1e-9)\n\ndef alpha_adaptativo(R, c):\n    """α_u = n_u / (n_u + c): mais avaliações -> mais peso no colaborativo."""\n    n = R.getnnz(axis=1)\n    return (n / (n + c)).astype(np.float32)[:, None]\n\ndef combinar(colab, conteudo, R, vistos, alpha=None, c=None, normalizar=True):\n    """Combina os dois scores. Use `alpha` (fixo) OU `c` (α adaptativo)."""\n    if normalizar:\n        colab, conteudo = minmax_por_usuario(colab, vistos), minmax_por_usuario(conteudo, vistos)\n    a = alpha_adaptativo(R, c) if c is not None else alpha\n    return a * colab + (1 - a) * conteudo\n\ndef score_hibrido(R, config):\n    """Híbrido completo a partir de uma configuração (dict) escolhida na validação."""\n    vistos = vistos_de(R)\n    colab = score_svd_sklearn(R, config[\'n_components\'])\n    conteudo = score_conteudo(R, ponderado=config[\'conteudo_ponderado\'])\n    return combinar(colab, conteudo, R, vistos,\n                    alpha=config.get(\'alpha\'), c=config.get(\'c\'))'
)

# Célula 55 — recomendador final
CELLS["recomendador final"] = (
    "def criar_recomendador(df, config):\n    \"\"\"Ajusta o híbrido em `df` e devolve uma função recomendar(user_id, n).\"\"\"\n    R = matriz(df)\n    vistos = vistos_de(R)\n    scores = score_hibrido(R, config)\n\n    def recomendar(user_id, n=10):\n        if user_id not in u_idx.index:\n            return f\"UserID {user_id} não encontrado.\"\n        u = u_idx[user_id]\n        # mesma rotina de top-K da avaliação (já exclui os filmes vistos)\n        top = top_k(scores[u:u + 1], vistos[u:u + 1], k=n)[0]\n        rec = movies.iloc[top][['MovieID', 'Title', 'Genres']].copy()\n        rec['Score'] = scores[u, top]\n        return rec.reset_index(drop=True)\n\n    return recomendar"
)


# Protocolo de avaliação (executado por `run_protocol`, não por `load`)
PROTOCOL_CELLS: dict[str, str] = {}

# Célula 21 — tuning do Surprise
PROTOCOL_CELLS["tuning do Surprise"] = (
    "from itertools import product\nfrom joblib import Parallel, delayed\nfrom surprise import Dataset, Reader, SVD, accuracy\n\nreader = Reader(rating_scale=(1, 5))\n\n# Ajuste SÓ no treino e escolha pelo RMSE na validação (o teste não participa)\ntrainset_tr = Dataset.load_from_df(train_df[['UserID', 'MovieID', 'Rating']], reader).build_full_trainset()\nvalidacao_surprise = list(val_df[['UserID', 'MovieID', 'Rating']].itertuples(index=False, name=None))\n\nparam_grid = {\n    'n_factors': [50, 100, 150],\n    'lr_all': [0.002, 0.005, 0.01],\n    'reg_all': [0.02, 0.05, 0.1],\n    'n_epochs': [20, 50, 70]\n}\ncombinacoes = [dict(zip(param_grid, v)) for v in product(*param_grid.values())]\n\ndef rmse_validacao(params):\n    modelo = SVD(**params, random_state=42)\n    modelo.fit(trainset_tr)\n    return accuracy.rmse(modelo.test(validacao_surprise), verbose=False)\n\nrmses = Parallel(n_jobs=-1)(delayed(rmse_validacao)(p) for p in combinacoes)\nbusca_surprise = pd.DataFrame(combinacoes).assign(**{'RMSE validação': rmses}) \\\n                   .sort_values('RMSE validação').reset_index(drop=True)\nmelhores_params = {k: (int(v) if k in ('n_factors', 'n_epochs') else float(v))\n                   for k, v in busca_surprise.iloc[0].drop('RMSE validação').items()}\n\nprint(f\"Melhores parâmetros (validação): {melhores_params}\")\nprint(f\"RMSE na validação: {busca_surprise['RMSE validação'].iloc[0]:.4f}\")"
)

# Célula 47 — fase de validação
PROTOCOL_CELLS["fase de validação"] = (
    "# ------------------------------------------------------------\n# FASE DE VALIDAÇÃO — ajuste no treino, comparação na validação\n# ------------------------------------------------------------\nR_tr = matriz(train_df)\nvistos_tr = vistos_de(R_tr)\nrel_val = relevantes_de(val_df)\nprint(f\"Usuários com ao menos 1 filme relevante na validação: {len(rel_val):,}\")\n\nvalidacao = [\n    avaliar(score_aleatorio(R_tr), 'Aleatório', vistos_tr, rel_val),\n    avaliar(score_popularidade(R_tr), 'Popularidade', vistos_tr, rel_val),\n]\n\n# Número de componentes do SVD sklearn, escolhido pela métrica de ranking\ncolab_val = {}\nfor k_comp in [20, 50, 100, 150, 200]:\n    colab_val[k_comp] = score_svd_sklearn(R_tr, k_comp)\n    validacao.append(avaliar(colab_val[k_comp], f'SVD sklearn ({k_comp} comp.)', vistos_tr, rel_val))\n\n# Variantes de conteúdo\nconteudo_val = {p: score_conteudo(R_tr, ponderado=p) for p in (False, True)}\nvalidacao.append(avaliar(conteudo_val[False], 'Conteúdo (original)', vistos_tr, rel_val))\nvalidacao.append(avaliar(conteudo_val[True], 'Conteúdo (ponderado por nota)', vistos_tr, rel_val))\n\ntabela_val = pd.DataFrame(validacao)\nsvd_val = tabela_val[tabela_val['Modelo'].str.startswith('SVD sklearn')]\nn_comp_top = int(svd_val.loc[svd_val[f'NDCG@{K}'].idxmax(), 'Modelo'].split('(')[1].split()[0])\nprint(f\"Componentes escolhidos para o SVD sklearn (NDCG@{K} na validação): {n_comp_top}\")\ntabela_val.round(4)"
)

# Célula 49 — híbrido na validação
PROTOCOL_CELLS["híbrido na validação"] = (
    "# ------------------------------------------------------------\n# Híbrido na validação: α fixo de 0 a 1 e α adaptativo com várias constantes c\n# ------------------------------------------------------------\nimport matplotlib.pyplot as plt\n\ncolab = colab_val[n_comp_top]\nnomes_conteudo = {False: 'Conteúdo original', True: 'Conteúdo ponderado por nota'}\n\nbusca_hibrido = []\nfor ponderado, nome in nomes_conteudo.items():\n    for a in np.round(np.arange(0, 1.01, 0.1), 1):\n        r = avaliar(combinar(colab, conteudo_val[ponderado], R_tr, vistos_tr, alpha=a),\n                    nome, vistos_tr, rel_val)\n        busca_hibrido.append({**r, 'tipo': 'fixo', 'alpha': a, 'c': None, 'ponderado': ponderado})\n    for c in [5, 10, 15, 25, 50, 100]:\n        r = avaliar(combinar(colab, conteudo_val[ponderado], R_tr, vistos_tr, c=c),\n                    nome, vistos_tr, rel_val)\n        busca_hibrido.append({**r, 'tipo': 'adaptativo', 'alpha': None, 'c': c, 'ponderado': ponderado})\nbusca_hibrido = pd.DataFrame(busca_hibrido)\n\n# Gráfico: α fixo na validação\nAZUL, LARANJA, CINZA, TEXTO, EIXO = '#2a78d6', '#eb6834', '#b4b2a9', '#52514e', '#c3c2b7'\nfig, ax = plt.subplots(figsize=(9, 4.5))\nresumo = []\nfor ponderado, cor in [(False, AZUL), (True, LARANJA)]:\n    d = busca_hibrido[(busca_hibrido['tipo'] == 'fixo') & (busca_hibrido['ponderado'] == ponderado)]\n    ax.plot(d['alpha'], d[f'NDCG@{K}'], color=cor, linewidth=2, marker='o', markersize=5,\n            label=nomes_conteudo[ponderado])\n    melhor = d.loc[d[f'NDCG@{K}'].idxmax()]\n    ax.plot(melhor['alpha'], melhor[f'NDCG@{K}'], 'o', markersize=12,\n            markerfacecolor='none', markeredgecolor=cor, markeredgewidth=2)\n    resumo.append(f\"{nomes_conteudo[ponderado]}: melhor α = {melhor['alpha']:.1f} \"\n                  f\"(NDCG {melhor[f'NDCG@{K}']:.3f})\")\npop_val = tabela_val.loc[tabela_val['Modelo'] == 'Popularidade', f'NDCG@{K}'].item()\nax.axhline(pop_val, color=TEXTO, linestyle='--', linewidth=1)\nax.text(0.01, pop_val + 0.002, 'baseline popularidade', fontsize=9, color=TEXTO)\nax.text(0.15, 0.08, '\\n'.join(resumo), transform=ax.transAxes, fontsize=9, color=TEXTO)\nax.set_title(f'Validação: NDCG@{K} do híbrido conforme o peso α do colaborativo',\n             loc='left', fontsize=11, color=TEXTO)\nax.set_xlabel('α  (0 = só conteúdo · 1 = só colaborativo)', color=TEXTO)\nax.set_ylabel(f'NDCG@{K}', color=TEXTO)\nax.set_xticks(np.round(np.arange(0, 1.01, 0.1), 1))\nfor lado in ['top', 'right']:\n    ax.spines[lado].set_visible(False)\nfor lado in ['left', 'bottom']:\n    ax.spines[lado].set_color(EIXO)\nax.tick_params(colors=TEXTO)\nax.grid(axis='y', color='#e1e0d9', linewidth=0.8)\nax.legend(frameon=False, loc='center right')\nplt.tight_layout()\nplt.show()\n\n# α adaptativo: NDCG por constante c\nbusca_hibrido[busca_hibrido['tipo'] == 'adaptativo'] \\\n    .pivot(index='c', columns='Modelo', values=f'NDCG@{K}').round(4)"
)

# Célula 51 — configuração escolhida
PROTOCOL_CELLS["configuração escolhida"] = (
    "# ------------------------------------------------------------\n# Escolha da configuração final — feita SÓ com a validação\n# ------------------------------------------------------------\nmelhor = busca_hibrido.loc[busca_hibrido[f'NDCG@{K}'].idxmax()]\nconfig_final = {\n    'n_components': n_comp_top,\n    'conteudo_ponderado': bool(melhor['ponderado']),\n    'alpha': None if melhor['tipo'] == 'adaptativo' else float(melhor['alpha']),\n    'c': None if melhor['tipo'] == 'fixo' else int(melhor['c']),\n}\nndcg_val_escolhido = melhor[f'NDCG@{K}']\nndcg_val_svd = svd_val[f'NDCG@{K}'].max()\n\nprint(\"Configuração escolhida na validação:\")\nfor chave, valor in config_final.items():\n    print(f\"  {chave:<20} {valor}\")\nprint(f\"\\nNDCG@{K} na validação: híbrido {ndcg_val_escolhido:.4f} | SVD sklearn sozinho {ndcg_val_svd:.4f}\")"
)

# Célula 53 — fase de teste
PROTOCOL_CELLS["fase de teste"] = (
    "# ------------------------------------------------------------\n# FASE DE TESTE — re-treino no desenvolvimento (treino + validação),\n# avaliação ÚNICA no teste. Nenhuma escolha é feita aqui.\n# ------------------------------------------------------------\nR_dev = matriz(dev_df)\nvistos_dev = vistos_de(R_dev)\nrel_teste = relevantes_de(test_df)\nprint(f\"Usuários com ao menos 1 filme relevante no teste: {len(rel_teste):,}\")\n\ncolab_dev_100 = score_svd_sklearn(R_dev, 100)\nconteudo_dev_original = score_conteudo(R_dev, ponderado=False)\nscores_hibrido_teste = score_hibrido(R_dev, config_final)\n\nteste = [\n    avaliar(score_aleatorio(R_dev), 'Aleatório', vistos_dev, rel_teste),\n    avaliar(score_popularidade(R_dev), 'Popularidade', vistos_dev, rel_teste),\n    avaliar(score_svd_sklearn(R_dev, n_comp_top), f'SVD sklearn ({n_comp_top} comp.)', vistos_dev, rel_teste),\n    avaliar(score_surprise(dev_df, melhores_params), 'SVD Surprise (nota prevista)', vistos_dev, rel_teste),\n    avaliar(conteudo_dev_original, 'Conteúdo (original)', vistos_dev, rel_teste),\n    avaliar(score_conteudo(R_dev, ponderado=True), 'Conteúdo (ponderado por nota)', vistos_dev, rel_teste),\n    # Versão inicial do notebook, para referência: 100 comp., conteúdo original, sem normalizar\n    avaliar(combinar(colab_dev_100, conteudo_dev_original, R_dev, vistos_dev, alpha=0.5, normalizar=False),\n            'Híbrido inicial (α=0.5, sem normalizar)', vistos_dev, rel_teste),\n    avaliar(scores_hibrido_teste, 'Híbrido final (config. da validação)', vistos_dev, rel_teste),\n]\ntabela_teste = pd.DataFrame(teste).sort_values(f'NDCG@{K}', ascending=False).reset_index(drop=True)\n\ndados = tabela_teste.sort_values(f'NDCG@{K}')\ncores = [CINZA if m in ('Aleatório', 'Popularidade') else AZUL for m in dados['Modelo']]\nfig, ax = plt.subplots(figsize=(9, 5))\nbarras = ax.barh(dados['Modelo'], dados[f'NDCG@{K}'], color=cores, height=0.6)\nax.axvline(tabela_teste.loc[tabela_teste['Modelo'] == 'Popularidade', f'NDCG@{K}'].item(),\n           color=TEXTO, linestyle='--', linewidth=1)\nfor b in barras:\n    ax.text(b.get_width() + 0.002, b.get_y() + b.get_height() / 2,\n            f'{b.get_width():.3f}', va='center', fontsize=9, color=TEXTO)\nax.set_title(f'Teste: NDCG@{K} por modelo (cinza = baseline; tracejado = popularidade)',\n             loc='left', fontsize=11, color=TEXTO)\nax.set_xlabel(f'NDCG@{K} (maior é melhor)', color=TEXTO)\nfor lado in ['top', 'right']:\n    ax.spines[lado].set_visible(False)\nfor lado in ['left', 'bottom']:\n    ax.spines[lado].set_color(EIXO)\nax.tick_params(colors=TEXTO)\nax.grid(axis='x', color='#e1e0d9', linewidth=0.8)\nax.set_axisbelow(True)\nplt.tight_layout()\nplt.show()\n\ntabela_teste.round(4)"
)


def load(movies: pd.DataFrame, ratings: pd.DataFrame, top_k=None) -> dict:
    """Executa as células com `movies` e `ratings` e devolve o namespace resultante,
    como se o notebook tivesse sido rodado até o fim com esses dados.

    `top_k` substitui a função de ranking do notebook (a original fica em
    `top_k_original`). O pacote desempata scores iguais pela ordem do catálogo; o
    notebook desempata de forma arbitrária, que muda com o NumPy e o processador.
    Com a mesma regra de desempate, todo o resto pode ser comparado bit a bit.
    """
    namespace = {"movies": movies.copy(), "ratings": ratings.copy(), "np": np, "pd": pd}
    for code in CELLS.values():
        exec(textwrap.dedent(code), namespace)
    namespace["top_k_original"] = namespace["top_k"]
    if top_k is not None:
        namespace["top_k"] = top_k  # avaliar() e criar_recomendador() leem esta global
    return namespace


def run_protocol(namespace: dict, surprise_params: dict | None = None, tune_surprise: bool = False) -> dict:
    """Roda as células de validação e teste sobre o namespace de `load`.

    Os gráficos das células são desenhados num backend sem janela e descartados.
    `surprise_params` substitui o resultado da busca do Surprise (81 combinações,
    lenta); com `tune_surprise=True`, a célula de busca roda de verdade.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    namespace.setdefault("plt", plt)
    show, plt.show = plt.show, lambda *args, **kwargs: None  # sem janela para mostrar
    try:
        for desc, code in PROTOCOL_CELLS.items():
            if desc == "tuning do Surprise":
                if not tune_surprise:
                    namespace["melhores_params"] = surprise_params
                    continue
            exec(textwrap.dedent(code), namespace)
            plt.close("all")
    finally:
        plt.show = show
    return namespace

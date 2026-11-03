"""Constantes e configuração do modelo final.

Todos os valores vêm do notebook `notebooks/01_modelagem_e_avaliacao.ipynb`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODELS_DIR / "recommender.joblib"

RANDOM_STATE = 42

# Split cronológico por usuário (notebook, seção "Separação em Treino, Validação e Teste")
TEST_FRAC = 0.2  # 20% mais recentes de cada usuário -> teste
VAL_FRAC = 0.2  # 20% mais recentes do que sobrou -> validação

# Avaliação top-N
K = 10
RELEVANCE_THRESHOLD = 4  # nota >= 4 conta como "o usuário gostou"


@dataclass(frozen=True)
class HybridConfig:
    """Configuração do recomendador híbrido.

    Use `alpha` (peso fixo do colaborativo) OU `c` (α adaptativo n/(n+c)), não os dois.
    """

    n_components: int = 20
    content_weighted: bool = True
    alpha: float | None = 0.7
    c: float | None = None

    def __post_init__(self) -> None:
        if (self.alpha is None) == (self.c is None):
            raise ValueError("Defina exatamente um entre `alpha` e `c`.")
        if self.alpha is not None and not 0 <= self.alpha <= 1:
            raise ValueError("`alpha` deve estar entre 0 e 1.")
        if self.n_components < 1:
            raise ValueError("`n_components` deve ser positivo.")


# Configuração escolhida na validação (notebook, seção "Configuração escolhida").
FINAL_CONFIG = HybridConfig(n_components=20, content_weighted=True, alpha=0.7, c=None)

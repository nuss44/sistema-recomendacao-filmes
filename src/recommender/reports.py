"""Relatórios da avaliação: metrics.json (fonte única das métricas), metrics.md e gráficos."""

from __future__ import annotations

import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import K

# Cores do notebook (validadas para daltonismo e contraste)
BLUE, ORANGE, GRAY, TEXT, AXIS, GRID = "#2a78d6", "#eb6834", "#b4b2a9", "#52514e", "#c3c2b7", "#e1e0d9"
BASELINES = ("Aleatório", "Popularidade")


# ----------------------------------------------------------------------
# JSON
# ----------------------------------------------------------------------
def _clean(value):
    """Converte tipos do numpy/pandas para JSON (NaN vira null)."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) else float(value)
    return value


def records(table: pd.DataFrame) -> list[dict]:
    return [_clean(row) for row in table.to_dict(orient="records")]


def environment() -> dict:
    """Versões que influenciam o resultado (o SVD é randomizado e depende do BLAS)."""
    from importlib.metadata import PackageNotFoundError, version

    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("recommender", "numpy", "pandas", "scipy", "scikit-learn", "scikit-surprise"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def write_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **payload}
    path.write_text(json.dumps(_clean(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# Markdown
# ----------------------------------------------------------------------
def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "não executado"
    return f"{value:.4f}".replace(".", ",")


def markdown_table(table: pd.DataFrame, k: int = K) -> str:
    columns = ["Modelo", f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}", "Cobertura"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join([str(row["Modelo"])] + [_fmt(row.get(c)) for c in columns[1:]]) + " |")
    return "\n".join(lines)


def write_markdown(validation: pd.DataFrame, test: pd.DataFrame, summary: list[str], path: Path) -> Path:
    text = "\n\n".join(
        [
            "# Métricas",
            "> Arquivo gerado por `python scripts/evaluate.py`. Não edite à mão.",
            "\n".join(summary),
            "## Teste (avaliação única, modelos re-treinados em treino + validação)",
            markdown_table(test),
            "## Validação (usada para as escolhas)",
            markdown_table(validation),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------
def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT)
    ax.set_axisbelow(True)


def plot_alpha_search(hybrid_search: pd.DataFrame, validation: pd.DataFrame, path: Path, k: int = K) -> Path:
    """NDCG na validação conforme o peso α do colaborativo, para as duas variantes de conteúdo."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {False: "Conteúdo original", True: "Conteúdo ponderado por nota"}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for weighted, color in ((False, BLUE), (True, ORANGE)):
        d = hybrid_search[(hybrid_search["tipo"] == "fixo") & (hybrid_search["ponderado"] == weighted)]
        best = d.loc[d[f"NDCG@{k}"].idxmax()]
        # O melhor α vai na própria legenda: não disputa espaço com as curvas.
        label = f"{names[weighted]} · melhor α = {best['alpha']:.1f} (NDCG {best[f'NDCG@{k}']:.3f})"
        ax.plot(d["alpha"], d[f"NDCG@{k}"], color=color, linewidth=2, marker="o", markersize=5, label=label)
        ax.plot(
            best["alpha"],
            best[f"NDCG@{k}"],
            "o",
            markersize=12,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=2,
        )
    popularity = validation.loc[validation["Modelo"] == "Popularidade", f"NDCG@{k}"].item()
    ax.axhline(popularity, color=TEXT, linestyle="--", linewidth=1)
    ax.annotate(
        "baseline popularidade",
        xy=(0, popularity),
        xytext=(4, -4),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=9,
        color=TEXT,
    )
    ax.set_title(
        f"Validação: NDCG@{k} do híbrido conforme o peso α do colaborativo", loc="left", fontsize=11, color=TEXT, pad=12
    )
    ax.set_xlabel("α  (0 = só conteúdo · 1 = só colaborativo)", color=TEXT)
    ax.set_ylabel(f"NDCG@{k}", color=TEXT)
    ax.set_ylim(bottom=0)
    ax.set_xticks([float(a) for a in np.round(np.arange(0, 1.01, 0.1), 1)])
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    _style(ax)
    return _save(fig, path)


def plot_test_ndcg(test: pd.DataFrame, path: Path, k: int = K) -> Path:
    """NDCG no teste por modelo; baselines em cinza, popularidade como linha de referência."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = test.dropna(subset=[f"NDCG@{k}"]).sort_values(f"NDCG@{k}")
    skipped = test.loc[test[f"NDCG@{k}"].isna(), "Modelo"].tolist()
    colors = [GRAY if m in BASELINES else BLUE for m in data["Modelo"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(data["Modelo"], data[f"NDCG@{k}"], color=colors, height=0.6)
    ax.axvline(test.loc[test["Modelo"] == "Popularidade", f"NDCG@{k}"].item(), color=TEXT, linestyle="--", linewidth=1)
    for bar in bars:
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}",
            va="center",
            fontsize=9,
            color=TEXT,
        )
    title = f"Teste: NDCG@{k} por modelo (cinza = baseline; tracejado = popularidade)"
    if skipped:
        title += f"\nNão executado: {', '.join(skipped)}"
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=12)
    ax.set_xlabel(f"NDCG@{k} (maior é melhor)", color=TEXT)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    _style(ax)
    return _save(fig, path)


def _save(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path

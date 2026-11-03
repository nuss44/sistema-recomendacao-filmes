"""Baixa o MovieLens 1M, converte para CSV e valida o resultado.

Uso (a partir da raiz do projeto):
    python scripts/download_data.py           # baixa só se ainda não existir
    python scripts/download_data.py --force   # baixa e converte de novo
    python scripts/download_data.py --data-dir outra/pasta

Saída: data/processed/{movies,users,ratings}.csv, com os mesmos nomes de
coluna usados no notebook de experimentos.
"""

from __future__ import annotations

import argparse
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

ROOT = Path(__file__).resolve().parents[1]

# Arquivos .dat do MovieLens: separador "::", sem cabeçalho, codificação latin-1.
# Zip-code é texto (ex.: "02460", "98107-2117"); ler como número perderia zeros à esquerda.
SCHEMAS: dict[str, dict] = {
    "movies": {"columns": ["MovieID", "Title", "Genres"], "dtype": {"MovieID": "int64"}},
    "users": {
        "columns": ["UserID", "Gender", "Age", "Occupation", "Zip-code"],
        "dtype": {"UserID": "int64", "Age": "int64", "Occupation": "int64", "Zip-code": "string"},
    },
    "ratings": {
        "columns": ["UserID", "MovieID", "Rating", "Timestamp"],
        "dtype": {"UserID": "int64", "MovieID": "int64", "Rating": "int64", "Timestamp": "int64"},
    },
}

# Contagens oficiais do MovieLens 1M (README do GroupLens).
EXPECTED_ROWS = {"movies": 3_883, "users": 6_040, "ratings": 1_000_209}


def show(path: Path) -> str:
    """Caminho curto para mensagens: relativo ao projeto quando possível."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def fmt(n: int) -> str:
    """Número no formato brasileiro: 1000209 -> '1.000.209'."""
    return f"{n:,}".replace(",", ".")


def ssl_context() -> ssl.SSLContext:
    """Contexto HTTPS com os certificados do pacote certifi, quando instalado.

    No Windows, o Python nem sempre encontra os certificados raiz do sistema e o
    download falha com CERTIFICATE_VERIFY_FAILED. O certifi traz o próprio
    conjunto de certificados e resolve isso sem desligar a verificação.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def download(url: str, destination: Path, force: bool = False) -> Path:
    """Baixa `url` para `destination`, a menos que o arquivo já exista."""
    if destination.exists() and not force:
        print(f"[ok] {show(destination)} já existe, download ignorado")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"[..] Baixando {url}")
    with urllib.request.urlopen(url, timeout=60, context=ssl_context()) as response, open(temporary, "wb") as file:
        while chunk := response.read(1 << 20):
            file.write(chunk)
    # Só renomeia no fim: um download interrompido nunca deixa um zip "válido" pela metade.
    temporary.replace(destination)
    print(f"[ok] Salvo em {show(destination)}")
    return destination


def extract(zip_path: Path, destination: Path) -> Path:
    """Extrai o zip e devolve a pasta com os arquivos .dat."""
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)
    return destination / "ml-1m"


def read_dat(path: Path, columns: list[str], dtype: dict) -> pd.DataFrame:
    """Lê um arquivo .dat do MovieLens."""
    return pd.read_csv(
        path,
        sep="::",
        engine="python",  # separadores com mais de um caractere exigem o engine Python
        header=None,
        names=columns,
        dtype=dtype,
        encoding="latin-1",
    )


def convert(dat_dir: Path) -> dict[str, pd.DataFrame]:
    """Lê os três arquivos .dat e devolve os DataFrames."""
    return {name: read_dat(dat_dir / f"{name}.dat", **schema) for name, schema in SCHEMAS.items()}


def validate(tables: dict[str, pd.DataFrame], expected_rows: dict[str, int] = EXPECTED_ROWS) -> list[str]:
    """Confere o conteúdo e devolve a lista de problemas (vazia = tudo certo)."""
    problems = []
    for name, expected in expected_rows.items():
        found = len(tables[name])
        if found != expected:
            problems.append(f"{name}: {fmt(found)} linhas, esperado {fmt(expected)}")

    for name, table in tables.items():
        if table.isna().any().any():
            problems.append(f"{name}: contém valores ausentes")

    movies, users, ratings = tables["movies"], tables["users"], tables["ratings"]
    if movies["MovieID"].duplicated().any():
        problems.append("movies: MovieID duplicado")
    if users["UserID"].duplicated().any():
        problems.append("users: UserID duplicado")
    if ratings.duplicated(["UserID", "MovieID"]).any():
        problems.append("ratings: o mesmo usuário avaliou o mesmo filme mais de uma vez")
    if not ratings["Rating"].between(1, 5).all():
        problems.append("ratings: notas fora do intervalo 1–5")
    if not ratings["UserID"].isin(users["UserID"]).all():
        problems.append("ratings: há UserID sem cadastro em users")
    if not ratings["MovieID"].isin(movies["MovieID"]).all():
        problems.append("ratings: há MovieID sem cadastro em movies")
    return problems


def save(tables: dict[str, pd.DataFrame], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = destination / f"{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8")
        print(f"[ok] {show(path)}: {fmt(len(table))} linhas")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="baixa e converte mesmo que os arquivos já existam")
    parser.add_argument("--url", default=URL, help="endereço alternativo do ml-1m.zip (ex.: um espelho)")
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data", help="pasta base (raw/ e processed/ ficam dentro)"
    )
    args = parser.parse_args(argv)
    raw_dir, processed_dir = args.data_dir / "raw", args.data_dir / "processed"

    targets = [processed_dir / f"{name}.csv" for name in SCHEMAS]
    if all(path.exists() for path in targets) and not args.force:
        print(f"[ok] CSVs já existem em {show(processed_dir)}. Use --force para refazer.")
        return 0

    try:
        zip_path = download(args.url, raw_dir / "ml-1m.zip", force=args.force)
    except OSError as error:
        print(f"[erro] Falha no download: {error}", file=sys.stderr)
        if "CERTIFICATE_VERIFY_FAILED" in str(error):
            print("       Instale os certificados e tente de novo: python -m pip install certifi", file=sys.stderr)
        print(f"       Baixe manualmente {URL} e salve em {show(raw_dir / 'ml-1m.zip')}", file=sys.stderr)
        return 1

    tables = convert(extract(zip_path, raw_dir))

    problems = validate(tables, EXPECTED_ROWS)
    if problems:
        print("[erro] Os dados não conferem com o MovieLens 1M oficial:", file=sys.stderr)
        for problem in problems:
            print(f"       - {problem}", file=sys.stderr)
        return 1

    save(tables, processed_dir)
    print("[ok] Dados prontos e validados.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

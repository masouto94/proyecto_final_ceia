"""Muestreo estratificado de Amazon Reviews'23 vía streaming HTTP, sin persistir el raw.

Se evita a propósito `datasets.load_dataset(..., streaming=True)`: al inferir el esquema Arrow
de a chunks, algunas categorías (p. ej. metadata de `Software`, que agrega campos como
`author`/`subtitle` ausentes en otras categorías) rompen el cast con
`CastError: ... because column names don't match`. Leer con `requests` + `json.loads` fila a
fila evita depender de un esquema homogéneo entre categorías.
"""

import json
import random
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import hf_hub_url

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
REVIEWS_PATH_TMPL = "raw/review_categories/{category}.jsonl"
META_PATH_TMPL = "raw/meta_categories/meta_{category}.jsonl"


def stream_jsonl(repo_path: str):
    """Itera línea a línea un .jsonl del repo vía HTTP streaming, sin persistir nada en disco."""
    url = hf_hub_url(REPO_ID, repo_path, repo_type="dataset")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                yield json.loads(line)


def stratified_reservoir_sample_reviews(category: str, reservoir_size: int, seed: int) -> pd.DataFrame:
    """Reservoir sampling (Algoritmo R) independiente por cada valor de rating (1-5)."""
    rng = random.Random(seed)
    reservoirs = {r: [] for r in range(1, 6)}
    seen = {r: 0 for r in range(1, 6)}

    for row in stream_jsonl(REVIEWS_PATH_TMPL.format(category=category)):
        rating = int(row["rating"])
        if rating not in reservoirs:
            continue
        seen[rating] += 1
        reservoir = reservoirs[rating]
        if len(reservoir) < reservoir_size:
            reservoir.append(row)
        else:
            j = rng.randint(0, seen[rating] - 1)
            if j < reservoir_size:
                reservoir[j] = row

    rows = [row for reservoir in reservoirs.values() for row in reservoir]
    df = pd.DataFrame(rows)
    df["category"] = category
    return df


def metadata_for_sampled_asins(category: str, asins: set) -> pd.DataFrame:
    """Filtra, en una sola pasada streaming, solo los parent_asin presentes en la muestra de reviews."""
    rows = [
        row
        for row in stream_jsonl(META_PATH_TMPL.format(category=category))
        if row["parent_asin"] in asins
    ]
    df = pd.DataFrame(rows)
    df["category"] = category
    return df


def build_stratified_sample(
    categories: list[str],
    reservoir_size_per_stratum: int,
    seed: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Corre el pipeline completo (reviews + metadata join-consistent) y exporta a Parquet.

    Devuelve las rutas de los dos Parquet generados.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reviews_samples = []
    for category in categories:
        df = stratified_reservoir_sample_reviews(category, reservoir_size_per_stratum, seed)
        print(f"{category}: muestreadas {len(df):,} reviews")
        reviews_samples.append(df)
    reviews_df = pd.concat(reviews_samples, ignore_index=True)

    meta_samples = []
    for category in categories:
        asins = set(reviews_df.loc[reviews_df["category"] == category, "parent_asin"])
        df = metadata_for_sampled_asins(category, asins)
        print(f"{category}: {len(asins):,} asins buscados, {len(df):,} encontrados en metadata")
        meta_samples.append(df)
    meta_df = pd.concat(meta_samples, ignore_index=True)

    reviews_path = output_dir / "reviews_sample.parquet"
    meta_path = output_dir / "meta_sample.parquet"
    reviews_df.to_parquet(reviews_path, index=False)
    meta_df.to_parquet(meta_path, index=False)

    return reviews_path, meta_path

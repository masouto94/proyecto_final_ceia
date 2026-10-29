"""Muestreo estratificado de Amazon Reviews'23 vía streaming HTTP, sin persistir el raw.

Se evita a propósito `datasets.load_dataset(..., streaming=True)`: al inferir el esquema Arrow
de a chunks, algunas categorías (p. ej. metadata de `Software`, que agrega campos como
`author`/`subtitle` ausentes en otras categorías) rompen el cast con
`CastError: ... because column names don't match`. Leer con `requests` + `json.loads` fila a
fila evita depender de un esquema homogéneo entre categorías.
"""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import HfApi, hf_hub_url

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


def remote_file_sizes(repo_paths: list[str]) -> dict[str, int]:
    """Tamaño en bytes de cada archivo del repo (sin bajarlo)."""
    infos = HfApi().get_paths_info(REPO_ID, repo_paths, repo_type="dataset")
    return {info.path: info.size for info in infos}


def byte_ranges(size: int, batch_bytes: int) -> list[tuple[int, int]]:
    """Parte [0, size) en rangos consecutivos de `batch_bytes`."""
    return [(start, min(start + batch_bytes, size)) for start in range(0, size, batch_bytes)]


def stream_jsonl_range(repo_path: str, start: int, end: int):
    """Itera (offset, fila) de las líneas que *empiezan* en el rango de bytes [start, end).

    Usa HTTP Range, así un batch se puede (re)procesar sin leer el archivo desde el principio.
    Pide desde `start - 1` y descarta hasta el primer `\\n`: si el byte anterior a `start` es un
    salto de línea, la línea que arranca en `start` es de este batch; si no, ese pedazo es la
    cola de una línea del batch anterior. Lee pasado `end` solo para terminar la última línea.
    El offset es la posición absoluta en el archivo: no depende del tamaño de batch.
    """
    url = hf_hub_url(REPO_ID, repo_path, repo_type="dataset")
    fetch_from = max(start - 1, 0)
    with requests.get(url, headers={"Range": f"bytes={fetch_from}-"}, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        if fetch_from > 0 and resp.status_code != 206:
            raise RuntimeError(f"El servidor ignoró el header Range para {repo_path}")

        skip_first = start > 0
        pos = fetch_from  # offset absoluto del primer byte de `buf`
        buf = b""
        for chunk in resp.iter_content(chunk_size=1 << 20):
            parts = (buf + chunk).split(b"\n")
            buf = parts.pop()
            for line in parts:
                line_start = pos
                pos += len(line) + 1
                if skip_first:
                    skip_first = False
                    continue
                if line_start >= end:
                    return
                if line.strip():
                    yield line_start, json.loads(line)
        # última línea del archivo, sin `\n` final
        if buf.strip() and not skip_first and pos < end:
            yield pos, json.loads(buf)


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


HASH_BUCKETS = 10_000

# (etiqueta, mínimo, máximo inclusive; None = sin tope) de #reviews totales del usuario
DEFAULT_ACTIVITY_TIERS = [("1", 1, 1), ("2-4", 2, 4), ("5+", 5, None)]


def user_in_hash_sample(user_id: str, fraction: float, salt: str = "") -> bool:
    """Decide si el usuario entra, de forma determinística y consistente entre categorías.

    Usa md5 (no `hash()`, que cambia entre procesos) para que el mismo `user_id` dé la misma
    decisión en cualquier corrida, categoría o máquina.
    """
    digest = hashlib.md5(f"{salt}{user_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % HASH_BUCKETS < fraction * HASH_BUCKETS


def _tier_of(n_reviews: int, tiers: list[tuple[str, int, int | None]]) -> str | None:
    for label, low, high in tiers:
        if n_reviews >= low and (high is None or n_reviews <= high):
            return label
    return None


def user_quota_sample_reviews(
    categories: list[str],
    quota_per_tier: int,
    hash_fraction: float,
    seed: int,
    tiers: list[tuple[str, int, int | None]] = DEFAULT_ACTIVITY_TIERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Muestreo por usuario: elige usuarios y se queda con **todas** sus reviews.

    1. Filtro por hash del `user_id` (`hash_fraction`): acota la memoria del conteo y es
       consistente entre categorías, así el historial cross-categoría queda completo.
    2. Pasada 1: cuenta reviews por usuario (sumando todas las `categories`).
    3. Cuota: hasta `quota_per_tier` usuarios al azar por tier de actividad.
    4. Pasada 2: junta todas las reviews de los usuarios elegidos.

    Devuelve (reviews, tiers_df). `tiers_df` trae el tamaño de cada tier en la población
    hasheada y el peso (`weight` = población / elegidos) para reponderar estadísticas.
    """
    user_counts = Counter()
    for category in categories:
        for row in stream_jsonl(REVIEWS_PATH_TMPL.format(category=category)):
            if user_in_hash_sample(row["user_id"], hash_fraction):
                user_counts[row["user_id"]] += 1

    selected, tiers_df = select_users_by_tier(user_counts, quota_per_tier, seed, tiers)

    reviews_samples = []
    for category in categories:
        rows = [
            row
            for row in stream_jsonl(REVIEWS_PATH_TMPL.format(category=category))
            if row["user_id"] in selected
        ]
        df = pd.DataFrame(rows)
        df["category"] = category
        reviews_samples.append(df)
    reviews_df = pd.concat(reviews_samples, ignore_index=True)
    reviews_df["user_tier"] = reviews_df["user_id"].map(selected)
    reviews_df["user_weight"] = reviews_df["user_tier"].map(tiers_df["weight"])

    return reviews_df, tiers_df


def select_users_by_tier(
    user_counts: dict[str, int],
    quota_per_tier: int,
    seed: int,
    tiers: list[tuple[str, int, int | None]] = DEFAULT_ACTIVITY_TIERS,
) -> tuple[dict[str, str], pd.DataFrame]:
    """Elige hasta `quota_per_tier` usuarios al azar por tier de actividad.

    Devuelve ({user_id: tier}, tiers_df) con población, elegidos y peso por tier.
    """
    rng = random.Random(seed)
    users_by_tier = {label: [] for label, _, _ in tiers}
    for user_id, n in user_counts.items():
        tier = _tier_of(n, tiers)
        if tier is not None:
            users_by_tier[tier].append(user_id)

    selected, tier_rows = {}, []
    for label, users in users_by_tier.items():
        users.sort()  # orden estable antes de samplear, para que el seed sea reproducible
        chosen = users if len(users) <= quota_per_tier else rng.sample(users, quota_per_tier)
        selected.update({u: label for u in chosen})
        tier_rows.append({
            "tier": label,
            "users_population": len(users),
            "users_selected": len(chosen),
            "weight": len(users) / len(chosen) if chosen else float("nan"),
        })
    tiers_df = pd.DataFrame(tier_rows).set_index("tier")
    return selected, tiers_df


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

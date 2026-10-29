"""Job batch del muestreo estratificado por (categoría, rating) + metadata join-consistent.

El Algoritmo R es secuencial (su estado depende de todo lo leído antes), así que no se puede
partir en batches independientes. Acá se usa *bottom-k*: cada review recibe una clave
pseudoaleatoria determinística (hash de seed + categoría + offset en el archivo) y por estrato
se quedan las `reservoir_size` claves más chicas. Es una muestra uniforme sin reemplazo, igual
que la del reservoir, pero cada batch se resuelve solo y el resultado se combina en el reduce:
reanudar desde el batch N da exactamente la misma muestra.

Etapas: reviews (bottom-k por batch) -> reduce (bottom-k global) -> meta.
Salidas: reviews_sample.parquet, meta_sample.parquet.
"""

import hashlib
import heapq
from dataclasses import dataclass

import pandas as pd

from proyecto_final_ceia.jobs.common import metadata_stage, plan_byte_batches
from proyecto_final_ceia.jobs.runner import Job, RunContext, Stage
from proyecto_final_ceia.sampling import REVIEWS_PATH_TMPL, stream_jsonl_range


@dataclass
class SampleStratifiedConfig:
    categories: list[str]
    reservoir_size: int = 2000  # filas por estrato (categoría, rating)
    seed: int = 42
    batch_mb: int = 64


def _sample_key(seed: int, category: str, offset: int) -> int:
    digest = hashlib.md5(f"{seed}|{category}|{offset}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1  # 63 bits: entra en int64 al pasar por Parquet


def _plan(config: SampleStratifiedConfig) -> list[dict]:
    return plan_byte_batches(REVIEWS_PATH_TMPL, config.categories, config.batch_mb)


def _bottom_k_batch(spec: dict, ctx: RunContext) -> pd.DataFrame:
    k = ctx.config.reservoir_size
    heaps = {r: [] for r in range(1, 6)}  # max-heap por clave negada: (-key, offset, row)
    for offset, row in stream_jsonl_range(spec["path"], spec["start"], spec["end"]):
        rating = int(row["rating"])
        if rating not in heaps:
            continue
        item = (-_sample_key(ctx.config.seed, spec["category"], offset), offset, row)
        heap = heaps[rating]
        if len(heap) < k:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)

    rows = [{**row, "_key": -neg_key} for heap in heaps.values() for neg_key, _, row in heap]
    df = pd.DataFrame(rows)
    df["category"] = spec["category"]
    return df


def _reduce_reviews(ctx: RunContext) -> None:
    k = ctx.config.reservoir_size
    samples = []
    for category in ctx.config.categories:  # de a una categoría para acotar memoria
        candidates = ctx.read_stage("reviews", where=lambda spec: spec["category"] == category)
        if candidates.empty:
            continue
        sample = candidates.sort_values("_key").groupby(candidates["rating"].astype(int), group_keys=False).head(k)
        samples.append(sample.drop(columns="_key"))
        ctx.log.info("reduce reviews %s: %d filas", category, len(sample))
    reviews = pd.concat(samples, ignore_index=True)
    reviews.to_parquet(ctx.run_dir / "reviews_sample.parquet", index=False)


JOB = Job(
    name="sample-stratified",
    description="Muestreo estratificado por (categoría, rating) con bottom-k reanudable + metadata.",
    config_cls=SampleStratifiedConfig,
    stages=[
        Stage("reviews", plan=_plan, process=_bottom_k_batch, reduce=_reduce_reviews),
        metadata_stage("reviews_sample.parquet", "meta_sample.parquet"),
    ],
)

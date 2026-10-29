"""Job batch de `user_quota_sample_reviews`: muestreo por usuario con cuota por tier de actividad.

Etapas: count_users (pasada 1) -> select (reduce) -> collect_reviews (pasada 2) -> meta.
Salidas: reviews_sample.parquet, meta_sample.parquet, tiers.parquet, selected_users.parquet.
"""

from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from proyecto_final_ceia.jobs.common import iter_batch_rows, metadata_stage, plan_byte_batches
from proyecto_final_ceia.jobs.runner import Job, RunContext, Stage
from proyecto_final_ceia.sampling import (
    DEFAULT_ACTIVITY_TIERS,
    REVIEWS_PATH_TMPL,
    select_users_by_tier,
    user_in_hash_sample,
)


@dataclass
class SampleUsersConfig:
    categories: list[str]
    quota_per_tier: int = 2000
    hash_fraction: float = 0.25
    seed: int = 42
    # [etiqueta, mínimo, máximo inclusive o null] de #reviews totales del usuario
    tiers: list[list] = field(default_factory=lambda: [list(t) for t in DEFAULT_ACTIVITY_TIERS])
    batch_mb: int = 64


def _plan_reviews(config: SampleUsersConfig) -> list[dict]:
    return plan_byte_batches(REVIEWS_PATH_TMPL, config.categories, config.batch_mb)


def _count_users(spec: dict, ctx: RunContext) -> pd.DataFrame:
    counts = Counter(
        row["user_id"]
        for row in iter_batch_rows(spec)
        if user_in_hash_sample(row["user_id"], ctx.config.hash_fraction)
    )
    return pd.DataFrame({"user_id": list(counts), "n": list(counts.values())})


def _select(ctx: RunContext) -> None:
    counts = ctx.read_stage("count_users").groupby("user_id")["n"].sum()
    tiers = [tuple(t) for t in ctx.config.tiers]
    selected, tiers_df = select_users_by_tier(counts.to_dict(), ctx.config.quota_per_tier, ctx.config.seed, tiers)
    pd.DataFrame({"user_id": list(selected), "user_tier": list(selected.values())}).to_parquet(
        ctx.run_dir / "selected_users.parquet", index=False
    )
    tiers_df.to_parquet(ctx.run_dir / "tiers.parquet")
    ctx.log.info("reduce select: %d usuarios hasheados, %d elegidos %s",
                 len(counts), len(selected), tiers_df["users_selected"].to_dict())


def _collect_reviews(spec: dict, ctx: RunContext) -> pd.DataFrame:
    if "selected" not in ctx.cache:
        ctx.cache["selected"] = set(pd.read_parquet(ctx.run_dir / "selected_users.parquet")["user_id"])
    selected = ctx.cache["selected"]
    df = pd.DataFrame([row for row in iter_batch_rows(spec) if row["user_id"] in selected])
    df["category"] = spec["category"]
    return df


def _reduce_reviews(ctx: RunContext) -> None:
    reviews = ctx.read_stage("collect_reviews")
    selected = pd.read_parquet(ctx.run_dir / "selected_users.parquet").set_index("user_id")["user_tier"]
    weights = pd.read_parquet(ctx.run_dir / "tiers.parquet")["weight"]
    reviews["user_tier"] = reviews["user_id"].map(selected)
    reviews["user_weight"] = reviews["user_tier"].map(weights)
    reviews.to_parquet(ctx.run_dir / "reviews_sample.parquet", index=False)
    ctx.log.info("reduce collect_reviews: %d reviews de %d usuarios", len(reviews), reviews["user_id"].nunique())


JOB = Job(
    name="sample-users",
    description="Muestreo por usuario (hash + cuota por tier de actividad) con historial completo y metadata.",
    config_cls=SampleUsersConfig,
    stages=[
        Stage("count_users", plan=_plan_reviews, process=_count_users, reduce=_select),
        Stage("collect_reviews", plan=_plan_reviews, process=_collect_reviews, reduce=_reduce_reviews),
        metadata_stage("reviews_sample.parquet", "meta_sample.parquet"),
    ],
)

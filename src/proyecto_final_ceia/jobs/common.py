"""Piezas compartidas por los jobs de sampling: planificación por rangos de bytes y metadata."""

import pandas as pd

from proyecto_final_ceia.jobs.runner import RunContext, Stage
from proyecto_final_ceia.sampling import META_PATH_TMPL, byte_ranges, remote_file_sizes, stream_jsonl_range

MB = 1 << 20


def plan_byte_batches(path_tmpl: str, categories: list[str], batch_mb: int) -> list[dict]:
    """Un batch por cada `batch_mb` de cada archivo; el total queda fijo antes de empezar."""
    paths = {category: path_tmpl.format(category=category) for category in categories}
    sizes = remote_file_sizes(list(paths.values()))
    missing = [category for category, path in paths.items() if path not in sizes]
    if missing:
        raise ValueError(f"No existen en el dataset: {missing} ({path_tmpl})")
    return [
        {"category": category, "path": path, "start": start, "end": end}
        for category, path in paths.items()
        for start, end in byte_ranges(sizes[path], batch_mb * MB)
    ]


def iter_batch_rows(spec: dict):
    """Filas del batch (sin offset)."""
    for _, row in stream_jsonl_range(spec["path"], spec["start"], spec["end"]):
        yield row


def metadata_stage(reviews_output: str, meta_output: str) -> Stage:
    """Etapa que filtra la metadata a los `parent_asin` presentes en `reviews_output`."""

    def asins_for(ctx: RunContext, category: str) -> set:
        if "asins" not in ctx.cache:
            reviews = pd.read_parquet(ctx.run_dir / reviews_output, columns=["category", "parent_asin"])
            ctx.cache["asins"] = reviews.groupby("category")["parent_asin"].agg(set).to_dict()
        return ctx.cache["asins"].get(category, set())

    def process(spec: dict, ctx: RunContext) -> pd.DataFrame:
        asins = asins_for(ctx, spec["category"])
        df = pd.DataFrame([row for row in iter_batch_rows(spec) if row["parent_asin"] in asins])
        df["category"] = spec["category"]
        return df

    def reduce(ctx: RunContext) -> None:
        meta = ctx.read_stage("meta")
        meta.to_parquet(ctx.run_dir / meta_output, index=False)
        ctx.log.info("reduce meta: %d productos -> %s", len(meta), meta_output)

    return Stage(
        name="meta",
        plan=lambda config: plan_byte_batches(META_PATH_TMPL, config.categories, config.batch_mb),
        process=process,
        reduce=reduce,
    )

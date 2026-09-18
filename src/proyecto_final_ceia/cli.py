"""CLI para correr el pipeline de sampling desde línea de comandos (pensado para el server).

Ejemplo:
    uv run sample-amazon-reviews \\
        --category Electronics --category Beauty_and_Personal_Care \\
        --reservoir-size 5000 --output-dir data/samples
"""

import argparse
from pathlib import Path

from proyecto_final_ceia.sampling import build_stratified_sample

DEFAULT_CATEGORIES = ["All_Beauty", "Amazon_Fashion", "Software"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category",
        dest="categories",
        action="append",
        help="Categoría a muestrear (repetible). Default: %(default)s",
        default=None,
    )
    parser.add_argument("--reservoir-size", type=int, default=2000, help="Filas por estrato (categoría, rating).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/samples"))
    args = parser.parse_args()

    categories = args.categories or DEFAULT_CATEGORIES

    reviews_path, meta_path = build_stratified_sample(
        categories=categories,
        reservoir_size_per_stratum=args.reservoir_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"reviews -> {reviews_path}")
    print(f"metadata -> {meta_path}")


if __name__ == "__main__":
    main()

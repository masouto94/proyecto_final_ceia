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


def ceia() -> None:
    """CLI único de jobs batch (reanudables, con manifest y log). Ver agents/WORKFLOW.md.

    uv run ceia jobs
    uv run ceia run sample-users --config configs/sample_users.json
    uv run ceia resume data/runs/sample-users_20261028T153012
    uv run ceia status data/runs/sample-users_20261028T153012
    """
    import json

    from proyecto_final_ceia.jobs import JOBS
    from proyecto_final_ceia.jobs.runner import DEFAULT_LOG_DIR, DEFAULT_RUNS_DIR, MANIFEST, load_config, resume_run, start_run

    parser = argparse.ArgumentParser(prog="ceia", description=ceia.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("jobs", help="Lista los jobs disponibles.")

    p_run = sub.add_parser("run", help="Arranca una corrida nueva.")
    p_run.add_argument("job", choices=sorted(JOBS))
    p_run.add_argument("--config", type=Path, required=True, help="JSON con los parámetros del job.")
    p_run.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p_run.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)

    p_resume = sub.add_parser("resume", help="Reanuda una corrida desde el primer batch que falta.")
    p_resume.add_argument("run_dir", type=Path)
    p_resume.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)

    p_status = sub.add_parser("status", help="Resume el manifest de una corrida.")
    p_status.add_argument("run_dir", type=Path)

    args = parser.parse_args()

    if args.command == "jobs":
        for name, job in sorted(JOBS.items()):
            print(f"{name:20s} {job.description}")
    elif args.command == "run":
        job = JOBS[args.job]
        start_run(job, load_config(job, args.config), args.runs_dir, args.log_dir)
    elif args.command == "resume":
        resume_run(JOBS, args.run_dir, args.log_dir)
    elif args.command == "status":
        m = json.loads((args.run_dir / MANIFEST).read_text(encoding="utf-8"))
        print(f"{m['run_id']}: {m['status']} | batches {m['batches']['done']}/{m['batches']['total']} "
              f"| creado {m['created_at']} | actualizado {m['updated_at']} | intentos {len(m['attempts'])}")
        if m["error"]:
            print(f"error: {m['error']}")
        if m["outputs"]:
            print(f"outputs: {m['outputs']}")


if __name__ == "__main__":
    main()

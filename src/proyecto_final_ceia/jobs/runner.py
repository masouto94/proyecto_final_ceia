"""Runner de jobs batch: run dir con manifest, logs livianos y reanudación por batch.

Un job es una lista de etapas. Cada etapa planifica sus batches (specs JSON-serializables), los
procesa de a uno escribiendo `stages/<etapa>/batch_<n>.parquet`, y opcionalmente cierra con un
`reduce`. Un batch está hecho si su Parquet existe (se escribe con rename atómico), así que
reanudar es saltear los que ya están y seguir desde el primero que falta.

Layout de una corrida:
    <runs_dir>/<job>_<timestamp>/
        manifest.json        # qué se corrió, cómo y cuándo (config, git, host, batches, estado)
        stages/<etapa>/...   # outputs intermedios por batch
        <resultados>         # lo que escriben los reduce
Log: <log_dir>/<job>_<timestamp>.log (uno por intento: run y cada resume).
"""

import dataclasses
import getpass
import json
import logging
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

DEFAULT_LOG_DIR = Path(os.environ.get("CEIA_LOG_DIR", "/var/log/ceia"))
DEFAULT_RUNS_DIR = Path(os.environ.get("CEIA_RUNS_DIR", "data/runs"))
MAX_RETRIES = 3
RETRY_WAIT_S = 30
MANIFEST = "manifest.json"


@dataclass
class Stage:
    name: str
    plan: Callable[[Any], list[dict]]
    process: Callable[[dict, "RunContext"], pd.DataFrame]
    reduce: Callable[["RunContext"], None] | None = None


@dataclass
class Job:
    name: str
    description: str
    config_cls: type
    stages: list[Stage]


@dataclass
class RunContext:
    run_dir: Path
    config: Any
    plan: dict[str, list[dict]]
    log: logging.Logger
    cache: dict = field(default_factory=dict)  # estado en memoria compartido entre batches

    def batch_path(self, stage: str, index: int) -> Path:
        return self.run_dir / "stages" / stage / f"batch_{index:05d}.parquet"

    def read_stage(self, stage: str, where: Callable[[dict], bool] = lambda spec: True) -> pd.DataFrame:
        """Concatena los Parquet de los batches de una etapa cuyo spec cumple `where`."""
        frames = [
            pd.read_parquet(self.batch_path(stage, i))
            for i, spec in enumerate(self.plan[stage])
            if where(spec)
        ]
        frames = [f for f in frames if len(f)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _fingerprint() -> dict:
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _setup_logger(job_name: str, log_dir: Path, timestamp: str) -> tuple[logging.Logger, Path]:
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{job_name}_{timestamp}.log"
        handler = logging.FileHandler(log_path, encoding="utf-8")
    except OSError as e:
        raise SystemExit(
            f"No se puede escribir el log en {log_dir} ({e}). Crearlo una vez: "
            f"`sudo mkdir -p {log_dir} && sudo chown $USER {log_dir}`, o usar --log-dir / CEIA_LOG_DIR."
        )
    log = logging.getLogger(f"ceia.{job_name}.{timestamp}")
    log.setLevel(logging.INFO)
    log.propagate = False
    for old in list(log.handlers):  # mismo job y segundo en el mismo proceso: no duplicar líneas
        old.close()
        log.removeHandler(old)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (handler, logging.StreamHandler(sys.stderr)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log, log_path


def _raise_on_sigterm(signum, frame):
    # `kill <pid>` sobre un proceso con nohup: convertirlo en excepción para dejar el manifest al día
    raise KeyboardInterrupt(f"señal {signum}")


def load_config(job: Job, config_path: Path):
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return job.config_cls(**raw)


def start_run(job: Job, config, runs_dir: Path, log_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    log, log_path = _setup_logger(job.name, log_dir, timestamp)
    try:
        plan = {stage.name: stage.plan(config) for stage in job.stages}
    except Exception:
        log.exception("ERROR planificando %s params=%s", job.name, json.dumps(dataclasses.asdict(config)))
        raise
    run_dir = Path(runs_dir) / f"{job.name}_{timestamp}"
    run_dir.mkdir(parents=True)
    manifest = {
        "job": job.name,
        "run_id": run_dir.name,
        "status": "created",
        "created_at": _now(),
        "updated_at": _now(),
        "finished_at": None,
        "config": dataclasses.asdict(config),
        "fingerprint": _fingerprint(),
        "batches": {"total": sum(len(specs) for specs in plan.values()), "done": 0},
        "plan": plan,
        "attempts": [],
        "error": None,
        "outputs": [],
    }
    _write_json(run_dir / MANIFEST, manifest)
    _execute(job, run_dir, log, log_path)
    return run_dir


def resume_run(jobs: dict[str, Job], run_dir: Path, log_dir: Path) -> Path:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / MANIFEST).read_text(encoding="utf-8"))
    if manifest["status"] == "done":
        raise SystemExit(f"{run_dir} ya terminó (status=done)")
    job = jobs[manifest["job"]]
    log, log_path = _setup_logger(job.name, log_dir, datetime.now().strftime("%Y%m%dT%H%M%S"))
    _execute(job, run_dir, log, log_path)
    return run_dir


def _execute(job: Job, run_dir: Path, log: logging.Logger, log_path: Path) -> None:
    manifest_path = run_dir / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = job.config_cls(**manifest["config"])
    plan = manifest["plan"]
    total = manifest["batches"]["total"]
    ctx = RunContext(run_dir=run_dir, config=config, plan=plan, log=log)

    flat = [(stage, i) for stage in job.stages for i in range(len(plan[stage.name]))]
    pending = [g for g, (stage, i) in enumerate(flat, start=1) if not ctx.batch_path(stage.name, i).exists()]
    resume_from = pending[0] if pending else None
    is_resume = bool(manifest["attempts"])

    manifest["status"] = "running"
    manifest["attempts"].append({
        "started_at": _now(),
        "log_file": str(log_path),
        "resumed_from_batch": resume_from if is_resume else None,
        "fingerprint": _fingerprint() if is_resume else None,
    })
    _write_json(manifest_path, manifest)

    log.info(
        "START %s run=%s total_batches=%d %sparams=%s",
        job.name, run_dir, total,
        f"resume_from_batch={resume_from} " if is_resume else "",
        json.dumps(manifest["config"], ensure_ascii=False),
    )

    signal.signal(signal.SIGTERM, _raise_on_sigterm)
    t_start = time.time()
    current = None
    try:
        g = 0
        for stage in job.stages:
            stage_specs = plan[stage.name]
            for i, spec in enumerate(stage_specs):
                g += 1
                out = ctx.batch_path(stage.name, i)
                if out.exists():
                    continue
                current = {"batch": g, "stage": stage.name, "stage_batch": i, "spec": spec}
                t0 = time.time()
                df = _with_retries(lambda: stage.process(spec, ctx), log, g)
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(".parquet.tmp")
                df.to_parquet(tmp, index=False)
                os.replace(tmp, out)
                manifest["batches"]["done"] = sum(ctx.batch_path(s.name, j).exists() for s, j in flat)
                manifest["updated_at"] = _now()
                _write_json(manifest_path, manifest)
                log.info("batch %d/%d done stage=%s %d/%d rows=%d %.0fs",
                         g, total, stage.name, i + 1, len(stage_specs), len(df), time.time() - t0)

            marker = run_dir / "stages" / stage.name / "_reduced"
            if stage.reduce is not None and not marker.exists():
                current = {"reduce": stage.name}
                stage.reduce(ctx)
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()

        manifest["status"] = "done"
        manifest["finished_at"] = _now()
        manifest["error"] = None
        manifest["outputs"] = sorted(p.name for p in run_dir.iterdir() if p.is_file() and p.name != MANIFEST)
        _write_json(manifest_path, manifest)
        log.info("DONE %s run=%s outputs=%s %.0fs", job.name, run_dir, manifest["outputs"], time.time() - t_start)
    except BaseException as e:
        interrupted = isinstance(e, KeyboardInterrupt)
        manifest["status"] = "interrupted" if interrupted else "failed"
        manifest["error"] = {"at": _now(), "where": current, "type": type(e).__name__, "message": str(e)}
        manifest["updated_at"] = _now()
        _write_json(manifest_path, manifest)
        if interrupted:
            log.error("INTERRUPTED en %s: %s", current, e)
        else:
            log.exception("ERROR en %s", current)
        log.error("reanudar con: uv run ceia resume %s", run_dir)
        raise


def _with_retries(fn: Callable[[], pd.DataFrame], log: logging.Logger, batch: int) -> pd.DataFrame:
    """Reintenta errores transitorios (red, HTTP) antes de dar el batch por fallido."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except (OSError, ConnectionError) as e:  # requests.RequestException hereda de OSError
            if attempt == MAX_RETRIES:
                raise
            log.warning("batch %d intento %d/%d falló (%s), reintento en %ds",
                        batch, attempt, MAX_RETRIES, e, RETRY_WAIT_S * attempt)
            time.sleep(RETRY_WAIT_S * attempt)

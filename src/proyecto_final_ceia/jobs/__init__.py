"""Registro de jobs batch. Para sumar uno: crear su módulo con un `JOB` y agregarlo acá."""

from proyecto_final_ceia.jobs import sample_stratified, sample_users

JOBS = {job.name: job for job in (sample_stratified.JOB, sample_users.JOB)}

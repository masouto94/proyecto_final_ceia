# Workflow y convenciones del proyecto

Ver [`SYSTEM_SPECS.md`](SYSTEM_SPECS.md) para el hardware detrás de estas decisiones.

## División del trabajo: desktop vs. server

1. **Todo lo exploratorio/experimental pasa primero por la PC de escritorio** (32GB RAM): EDA, prototipado de features, prueba de modelos, notebooks sueltos. Acá no hay que preocuparse tanto por memoria.
2. **Cuando un pipeline queda definido**, se refactoriza a código reusable en `src/proyecto_final_ceia/` (funciones puras, sin estado de notebook) pensado para correr con RAM acotada (streaming/chunks), y recién ahí se lleva al server.
3. El server nunca debería correr un experimento nuevo desde cero — corre versiones ya validadas.

## Estructura del repo

```
notebooks/          # exploración e iteración — pueden ser desprolijos, está bien
src/proyecto_final_ceia/  # lógica reusable, importable desde notebooks y desde el CLI
data/                # generado, nunca se versiona (gitignored)
agents/              # este directorio: specs y convenciones para vos y para Claude
```

- El paquete se instala en modo editable (`uv sync`, build backend `hatchling`, layout `src/`), así `from proyecto_final_ceia.sampling import ...` funciona igual desde un notebook que desde un script.
- Los notebooks numerados (`01_`, `02_`, `03_`...) son secuenciales: cada uno asume que el anterior ya corrió y dejó su output en `data/`.

## Cómo se consume el dataset (Amazon Reviews'23)

**Nunca bajar el raw completo.** Ni en el desktop (377GB no tiene sentido para experimentar) ni en el server (se come el disco entero sin dejar margen para Postgres/artefactos).

- **Conexión mínima / exploración de un `head()`**: `datasets.load_dataset("json", data_files="hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/<categoria>.jsonl", split="train")` — sin `streaming=True`, esto cachea el archivo localmente (`~/.cache/huggingface/hub`), sirve para inspección puntual de 1-2 categorías chicas.
- **Cualquier cosa que toque varias categorías o categorías grandes**: usar streaming manual con `requests` + `json.loads` línea por línea (`proyecto_final_ceia.sampling.stream_jsonl`), **no** `datasets.load_dataset(..., streaming=True)`. Este último infiere el esquema Arrow por chunks y rompe con `CastError` cuando una categoría tiene campos que otra no tiene (pasó con la metadata de `Software`, que agrega `author`/`subtitle`).
- **Muestreo representativo**: reservoir sampling (Algoritmo R) estratificado por `(categoría, rating)` vía `proyecto_final_ceia.sampling.stratified_reservoir_sample_reviews` — una sola pasada streaming, sin necesitar conocer de antemano el tamaño del archivo, sin persistir el raw.
- **Metadata join-consistent**: después de samplear reviews, se filtra la metadata solo para los `parent_asin` presentes en la muestra (`metadata_for_sampled_asins`) — evita bajar/samplear toda la metadata por separado.

## Correr el pipeline

Desde notebook (exploración, paso a paso, para inspeccionar resultados intermedios):
ver [`notebooks/02_stratified_sampling.ipynb`](../notebooks/02_stratified_sampling.ipynb).

Desde CLI (server, o cualquier corrida "de una"):
```bash
uv run sample-amazon-reviews \
    --category Electronics --category Beauty_and_Personal_Care \
    --reservoir-size 5000 --output-dir data/samples
```

## Reglas de memoria/disco a respetar en código nuevo

- No usar `describe(include="all")` (ni operaciones que requieran hashear) sobre columnas con listas/dicts (`images`, `features`, `description`, `videos`, `categories`, `details`, `bought_together`) — pandas se cuelga comparando objetos no-hasheables elemento a elemento. Separar esas columnas y resumirlas aparte (null count, largo promedio) — ver [`notebooks/03_describe_sample.ipynb`](../notebooks/03_describe_sample.ipynb).
- Cualquier función que vaya a correr en el server debe poder trabajar por partes (generadores, `streaming=True` propio, lectura de Parquet en batches) — nunca asumir que un `DataFrame` completo entra en 7,5GB de RAM.
- Los archivos bajo `data/` no se commitean nunca — están en `.gitignore` junto con `.ipynb_checkpoints/`.

## Al escribir/editar notebooks

Los notebooks creados con el `Write` tool necesitan un campo `"id"` explícito en cada celda (nbformat 4.5+). Sin eso, herramientas de lectura posicional pueden desincronizarse al insertar/borrar celdas y terminar editando la celda equivocada. Usar siempre IDs explícitos y descriptivos (`"id": "reviews-describe"`, no autogenerados) al crear un notebook nuevo.

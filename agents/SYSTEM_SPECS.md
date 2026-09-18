# System specs

Dos máquinas, dos roles. No mezclar el trabajo pesado/exploratorio con lo que corre en el server.

## PC de escritorio — exploración y experimentación

- RAM: 32 GB
- Buen ancho de banda, tanto subida como bajada
- Rol: EDA, feature engineering, prototipado de modelos, notebooks, cualquier cosa que necesite iterar rápido con datasets grandes en memoria
- Es donde se valida un pipeline antes de portarlo al server

## Server remoto "mefisto" — pipelines productivos

Notebook ASUS (Intel Tiger Lake-LP) corriendo Ubuntu, accedida por SSH/VPN. Detalle completo en `estado_servidor.md` (fuera de este repo); acá solo lo relevante para decisiones de diseño.

| Recurso | Valor | Implicancia |
|---|---|---|
| RAM | **7,5 GB** | Es el cuello de botella real. Nada de cargar el dataset completo (ni siquiera muestras grandes) en un solo `DataFrame` en memoria — todo lo que corra acá tiene que ser streaming/chunked. |
| Disco | 512 GB NVMe, **377 GB libres** en la raíz | Alcanza casi exacto para el raw completo del dataset (377GB), pero sin margen para Postgres/artefactos/checkpoints si se llena de raw. Nunca persistir el raw completo acá — solo muestras ya reducidas (Parquet) y artefactos de modelo. |
| GPU | NVIDIA RTX 3050 Ti Mobile, **4 GB VRAM** | Alcanza para fine-tuning de modelos chicos e inferencia cuantizada (~7B). No alcanza para entrenar transformers grandes desde cero — coincide con el alcance de "modelo dedicado" del charter, no con el "pre entrenado" a gran escala. |
| Kernel / driver | 6.8.0-139-generic, NVIDIA 595.91.07 (CUDA 13.2) | — |

### Limitaciones operativas a tener en cuenta

- **La BIOS es inalcanzable** (pantalla interna muerta, sin video por USB-C durante POST). Cualquier cambio que dependa de tocar el UEFI queda descartado.
- **El arranque pide la passphrase de LUKS desde la consola física.** El server no vuelve solo después de un corte de luz ni se puede reiniciar de forma desatendida — hay que estar físicamente para destrabarlo.
- Secure Boot activo sin poder desactivarse; los drivers NVIDIA que se instalen tienen que venir del canal `restricted` de Ubuntu (firmados por Canonical), no DKMS con MOK propio.
- Temperatura en reposo ~54°C — vigilar si vuelven timeouts de I/O.

## Dataset: Amazon Reviews'23 (`McAuley-Lab/Amazon-Reviews-2023`, HuggingFace)

- 34 categorías (33 + "Unknown"), **571,54M reviews** en total, **377 GB** entre `raw/review_categories/` (275GB) y `raw/meta_categories/` (101GB)
- El repo depende de un *loading script* (`Amazon-Reviews-2023.py`) que las versiones actuales de `datasets` (>=3.0) ya no soportan (`Dataset scripts are no longer supported`) — no usar `load_dataset(..., trust_remote_code=True)` tal como indica la guía oficial
- No hay export a Parquet auto-generado por HF para este dataset (demasiado grande / script custom) — no sirve `duckdb`/`polars` con `hf://.../refs/convert/parquet`
- Tabla completa de tamaños por categoría (reviews + metadata) y de `#Rating` (reviews) por categoría: ver historial del proyecto o recalcular con `huggingface_hub.HfApi().repo_info(..., files_metadata=True)`

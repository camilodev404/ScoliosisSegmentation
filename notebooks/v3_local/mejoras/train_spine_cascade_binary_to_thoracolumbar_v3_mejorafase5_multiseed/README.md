# Fase 5 — Robustez estadística / multiseed (`mejorafase5_multiseed`)

**Estado:** **cerrada** (2026-05-15). Tres runs **`_cuda`** (semillas 42, 1337, 4242) ejecutados. **Decisión global:** **«Prometedor; no consolidado»** — alta varianza del macro multiclase entre semillas; **sin cambios** en el notebook vivo (detalle en `../RESULTADOS_Y_DECISIONES_GENERAL.md` § Fase 5).

## Objetivo (plan §3 Fase 5)

Repetir el **mismo** procedimiento (mismo split, mismas épocas y augment ROI) con **2–3 semillas** distintas para ver si la mejora de `macro_dice_fg` / `macro_iou_fg` frente al baseline se **replica** en magnitud y dirección. Criterio: si solo una semilla gana → *prometedor, no consolidado*; si hay replicación → valorar adopción explícita en documentación.

## Qué cambia por variante

Solo `SEED` en la celda de reproducibilidad, `OUTPUT_DIR` bajo `outputs/analysis_outputs_v3/` y nombres de checkpoints (sufijo `s42`, `s1337`, `s4242`). No se modifica `seed: int = 42` en firmas de funciones auxiliares.

## Regenerar los notebooks

Desde la **raíz del repositorio**:

```text
python notebooks/v3_local/mejoras/scripts/build_fase5_multiseed_notebooks.py
```

Entrada: `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3.ipynb`.

Salida en esta carpeta:

- `train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase5_multiseed_cuda_seed42.ipynb`
- `train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase5_multiseed_cuda_seed1337.ipynb`
- `train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase5_multiseed_cuda_seed4242.ipynb`

Al **final** de cada notebook heredado del base figura el bloque **Registro de ejecución del notebook**; ejecutar esa celda al cerrar cada run.

Los tres `*_cuda_seed*.ipynb` **ejecutados** incluyen bloques de **Análisis e interpretación** (tabla de las tres semillas + nota por semilla). Tras **regenerar** con `build_fase5_multiseed_notebooks.py`, volver a aplicar:

```text
python notebooks/v3_local/mejoras/scripts/patch_fase5_multiseed_analysis_cells.py
```

*(O `restore_executed_cuda_analysis_cells.py` para todas las fases `_cuda` cerradas.)*

## Carpetas de salida (`outputs/analysis_outputs_v3/`)

- `training_runs_cascade_v3_fase5_multiseed_cuda_s42`
- `training_runs_cascade_v3_fase5_multiseed_cuda_s1337`
- `training_runs_cascade_v3_fase5_multiseed_cuda_s4242`

## Al cerrar la fase

*(Hecho, 2026-05-15)* `RESULTADOS_Y_DECISIONES_GENERAL.md` (§7.2 + sección Fase 5), `experiment_registry.csv`, celdas finales en los tres `*_cuda*.ipynb` (vía `patch_fase5_multiseed_analysis_cells.py`). Comprobar que exista `thoracolumbar_core_test_metrics.csv` en cada carpeta de salida (p. ej. en `s1337` se repuso si el run no lo había escrito en disco).

**Anterior:** Fase 4 — `../train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase4_augmentacion_roi/README.md`.

**Siguiente:** Fase 6 — `../train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase6_postproceso_ligero/README.md` (detalle en `../PLAN_ACCION_AJUSTES_MODELOS.md` §3).

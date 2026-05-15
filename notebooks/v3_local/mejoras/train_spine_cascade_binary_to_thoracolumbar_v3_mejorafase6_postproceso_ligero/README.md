# Fase 6 — Post-proceso ligero anatómico (`mejorafase6_postproceso_ligero`)

**Estado:** **cerrada** (2026-05-15). Variante **`_cuda`** ejecutada con `FASE6_VERTICAL_MEDIAN_K=0` e islas mínimas (`FASE6_MIN_ISLAND_PIXELS=64`). **Decisión:** **No adoptar** el post en el notebook vivo (detalle en `../RESULTADOS_Y_DECISIONES_GENERAL.md` § Fase 6).

## Resultado `_cuda` (resumen)

- **Salida:** `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase6_postproc_cuda/`
- **Multiclase test (sin post):** `macro_dice_fg` **0,2294** \| `macro_iou_fg` **0,1360**
- **Multiclase test (con post islas):** `macro_dice_fg` **0,2273** \| `macro_iou_fg` **0,1349** (Δ macro **−0,0021**)
- **Binario test:** Dice **0,8710** (inalterado por el post multiclase)
- **Torácico:** T9–T12 mayormente **bajan** con el post; no se evaluó `K=3` (no necesario para decidir).

## Objetivo (plan §3 Fase 6)

Post-proceso ligero en **test** multiclase: islas mínimas por etiqueta; mediana vertical opcional (`FASE6_VERTICAL_MEDIAN_K` impar ≥ 3).

## Regenerar los notebooks

```text
python notebooks/v3_local/mejoras/scripts/build_fase6_postproceso_notebooks.py
```

Tras regenerar, volver a insertar el análisis del run `_cuda` cerrado con:

```text
python notebooks/v3_local/mejoras/scripts/patch_fase6_postproceso_analysis_cells.py
```

*(O `restore_executed_cuda_analysis_cells.py`.) Los `_cpu` sin ejecutar conservan la sección vacía hasta su propio análisis.*

## Carpetas de salida

- `training_runs_cascade_v3_fase6_postproc_cuda`
- `training_runs_cascade_v3_fase6_postproc_cpu` (opcional)

## Al cerrar la fase

*(Hecho)* `RESULTADOS_Y_DECISIONES_GENERAL.md`, `experiment_registry.csv` y celdas de análisis en `…_postproceso_ligero_cuda.ipynb`.

**Anterior:** Fase 5 — `../train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase5_multiseed/README.md`.

**Siguiente:** Fase 7 — `../train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase7_auxiliares_rango_lastvis/README.md`.

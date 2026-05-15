# Fase 4 — Augmentación geométrica suave en ROI multiclase (`mejorafase4_augmentacion_roi`)

**Estado:** variante **`_cuda`** ejecutada y cerrada (2026-05-14). **Decisión:** **Adoptar** augment ROI en train (detalle en `RESULTADOS_Y_DECISIONES_GENERAL.md`). La misma configuración quedó **integrada** en `../train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` (`apply_roi_augment=True` en `multiclass_train_ds`).

## Resultado `_cuda` (resumen)

- **Salida:** `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase4_augment_roi_cuda/`
- **Multiclase test:** `macro_dice_fg` **0,2380** (baseline 0,1894; Fase 3 run B 0,2345); `macro_iou_fg` **0,1415** (baseline 0,1126; run B 0,1387).
- **Binario test:** Dice **0,8710** (baseline 0,8713).
- **Val:** mejor `val_macro_dice_fg` ≈ **0,2325** (época **22**).
- **ROI:** **198/198** `pred_binary`.

## Objetivo (plan §3 Fase 4)

Rotación y escala **pequeñas** en el recorte ROI del multiclase (**solo train**), acotadas para radiografías, sobre el cascada V3 **vivo** (Fase 3 integrada).

## Implementación

- **`apply_fase4_roi_geom_augment_uint8`** y flags en `prepare_multiclass_cascade_sample` / `CascadedThoracolumbarDataset` en el notebook vivo.
- Los notebooks de esta fase usan carpetas y checkpoints propios; al **regenerar** desde el vivo ya con `apply_roi_augment=True`, el script `build_fase4_augment_roi_notebooks.py` detecta train **ON** y no falla.

## Regenerar los notebooks (`_cpu` / `_cuda`)

Desde la **raíz del repositorio**:

```text
python notebooks/v3_local/mejoras/scripts/build_fase4_augment_roi_notebooks.py
```

Entrada: `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3.ipynb`.

Al **final del notebook** hay un bloque **Registro de ejecución del notebook** (tabla + celda de código): ejecutá esa celda al cerrar el run para dejar fecha/hora en la salida guardada.

**Nota:** al regenerar, el script **vuelve a poner** las celdas finales heredadas del base; si ya cerraste la fase, vuelve a pegar el análisis en las celdas **25–27** del `*_cuda.ipynb` o conserva una copia del notebook cerrado.

Salida en esta carpeta:

- `train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase4_augmentacion_roi_cpu.ipynb`
- `train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase4_augmentacion_roi_cuda.ipynb`

## Carpetas de salida (`outputs/analysis_outputs_v3/`)

- `training_runs_cascade_v3_fase4_augment_roi_cpu`
- `training_runs_cascade_v3_fase4_augment_roi_cuda`

## Al cerrar la fase

*(Hecho)* `RESULTADOS_Y_DECISIONES_GENERAL.md`, `experiment_registry.csv`, celdas finales del `*_cuda.ipynb`.

**Anterior:** Fase 3 — `../train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase3_scheduler_lr/README.md`.

**Siguiente:** Fase 5 multiseed **cerrada** (`../train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase5_multiseed/`). **Fase 6** **cerrada** — `../train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase6_postproceso_ligero/README.md`. **Fase 7** — `../train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase7_auxiliares_rango_lastvis/README.md` y `PLAN_ACCION_AJUSTES_MODELOS.md` §3.

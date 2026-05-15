# Fase 3 — Optimización multiclase: LR + scheduler (`mejorafase3_scheduler_lr`)

**Estado:** variante **`_cuda`** (run A) y **verificación run B** ejecutadas y cerradas (2026-05-15 / 2026-05-14). **Decisión global:** **Adoptar** LR multiclase ×0,5 + CosineAnnealingLR. **Integrado** en el notebook vivo `../train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` (misma lógica que experimentos Fase 3; rutas de salida por defecto `training_runs_cascade_v3/`). Detalle en `RESULTADOS_Y_DECISIONES_GENERAL.md`.

## Comprobación de reproducibilidad (run B) — cerrada

- **Notebook:** `train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase3_scheduler_lr_cuda_repro_verificacion.ipynb`
- **Métricas:** `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase3_scheduler_lr_cuda_repro_b/`
- **Lectura:** run B alinea **binario** con baseline (Dice **0,8710** vs baseline **0,8713**); `macro_dice_fg` **0,2345** (vs A **0,2323**, vs baseline **0,1894**). La bajada binaria solo en A se atribuye a variación GPU; condición de adopción **levantada**.
- **Registro:** fila `3_cuda_repro_b` en `experiment_registry.csv`; ficha §7.1 y tabla §7.2 actualizadas.

Para **regenerar** el notebook de verificación:

```text
python notebooks/v3_local/mejoras/scripts/build_fase3_scheduler_lr_cuda_repro_notebook.py
```

## Resultado `_cuda` (resumen)

- **Salida:** `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase3_scheduler_lr_cuda/`
- **Multiclase test:** `macro_dice_fg` **0,2323** (baseline 0,1894); `macro_iou_fg` **0,1375** (baseline 0,1126).
- **Binario test:** Dice **0,8611** (baseline 0,8713) — revisar con re-run antes de adopción plena.
- **Val:** mejor `val_macro_dice_fg` ≈ **0,2345** (época **22**).

## Objetivo (plan §3)

Aplicar **LR más bajo** en la etapa **multiclase** y **cosine annealing** (`CosineAnnealingLR` con `T_max=MULTICLASS_EPOCHS`) para estabilizar curvas y reducir sobreajuste a bordes de ROI. La etapa **binaria** sigue con `LR` sin scheduler en esta fase. El mejor checkpoint multiclase sigue eligiéndose por **mejor `val_macro_dice_fg`** (igual que el base).

## Regenerar los notebooks (`_cpu` / `_cuda`)

Desde la **raíz del repositorio**:

```text
python notebooks/v3_local/mejoras/scripts/build_fase3_scheduler_lr_notebooks.py
```

Entrada: `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` (sin letterbox ni pesos Fase 2).

Salida en esta carpeta:

- `train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase3_scheduler_lr_cpu.ipynb`
- `train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase3_scheduler_lr_cuda.ipynb`

Escalas por defecto: `LR_MULTICLASS = LR * 0,75` (`_cpu`), `LR * 0,5` (`_cuda`), editables en `VARIANTS` del script.

## Carpetas de salida (`outputs/analysis_outputs_v3/`)

- `training_runs_cascade_v3_fase3_scheduler_lr_cpu`
- `training_runs_cascade_v3_fase3_scheduler_lr_cuda` (run A, referencia principal ya ejecutada)
- `training_runs_cascade_v3_fase3_scheduler_lr_cuda_repro_b` (run B, solo si se ejecuta el notebook de verificación)

## Entradas (no mover)

Mismo manifiesto y dataset que el cascada V3; baseline en `outputs/analysis_outputs_v3/training_runs_cascade_v3/`.

## Al cerrar la fase

Actualizar `RESULTADOS_Y_DECISIONES_GENERAL.md` (ficha §7.1 + fila §7.2), `experiment_registry.csv` y las **celdas finales** del `*_cuda.ipynb` con métricas reales (misma convención que Fases 1–2).

**Anterior:** Fase 2 — `../train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase2_pesos_clases_t7_t12/README.md`.

**Siguiente:** Fase 4 — `../train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase4_augmentacion_roi/README.md` (**cerrada**). Fase 5 — ver `PLAN_ACCION_AJUSTES_MODELOS.md` §3.

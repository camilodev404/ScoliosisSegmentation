# Fase 2 — Pérdida / muestreo consciente de clase (T7–T12) (`mejorafase2_pesos_clases_t7_t12`)

**Estado:** variante **`_cuda`** ejecutada y cerrada (2026-05-14); decisión registrada en `RESULTADOS_Y_DECISIONES_GENERAL.md` y `experiment_registry.csv`: **No adoptar** el multiplicador CE **1,40×** en T7–T12 tal cual en el procedimiento base. La variante **`_cpu`** sigue siendo opcional.

## Objetivo (plan §3)

Reforzar el entrenamiento multiclase hacia **T7–T12** multiplicando los pesos de **cross-entropy** ya estimados (`estimate_multiclass_class_weights`) para `class_id` **7–12** (T7–T12), **antes** de `nn.CrossEntropyLoss`. El término **Dice** del mismo bucle no se modifica en esta fase.

## Regenerar los notebooks (`_cpu` / `_cuda`)

Desde la **raíz del repositorio**:

```text
python notebooks/v3_local/mejoras/scripts/build_fase2_pesos_t7_t12_notebooks.py
```

Entrada: `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3.ipynb`.

Salida en esta carpeta:

- `train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase2_pesos_clases_t7_t12_cpu.ipynb`
- `train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase2_pesos_clases_t7_t12_cuda.ipynb`

Multiplicadores por defecto (editables en `VARIANTS` del script de build): **1.25** (`_cpu`), **1.40** (`_cuda`).

## Carpetas de salida (bajo `outputs/analysis_outputs_v3/` en la raíz del repo)

- `training_runs_cascade_v3_fase2_pesos_t7_t12_cpu`
- `training_runs_cascade_v3_fase2_pesos_t7_t12_cuda`

Checkpoints: nombres `*_fase2_pesos_t7_t12_{cpu|cuda}_best.pt` (binario y multiclase), coherentes con el script.

## Entradas (no mover)

Mismo manifiesto, `data/Scoliosis_Dataset/`, baseline de referencia en `outputs/analysis_outputs_v3/training_runs_cascade_v3/` (rutas relativas a la raíz del repo, coherente con `OUTPUT_DIR` del notebook base).

## Resultado `_cuda` (resumen)

- **Salida:** `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase2_pesos_t7_t12_cuda/`
- **Multiclase test:** `macro_dice_fg` ≈ **0,1896** (baseline 0,1894); `macro_iou_fg` ≈ **0,1104** (baseline 0,1126).
- **Por clase:** mejora clara en **T7–T12**; regresión relevante en **L5** respecto al baseline.
- **Decisión:** **No adoptar** (detalle en `RESULTADOS_Y_DECISIONES_GENERAL.md` § Fase 2).

**Anterior:** Fase 1 — `../train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_letterbox_roi/README.md` (decisión: **No adoptar** letterbox en la configuración probada).

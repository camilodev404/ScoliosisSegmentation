# Fase 7 — Auxiliares y parciales / rango última vértebra visible (`mejorafase7_auxiliares_rango_lastvis`)

**Estado:** **`_cuda` completo** (2026-05-15) — cascada + last_visible + clipping cerrados. Decisión: **No adoptar** (ver `../RESULTADOS_Y_DECISIONES_GENERAL.md`).

## Objetivo (plan §3 Fase 7)

Notebook **autocontenido** (misma regla que Fases 1–6):

1. Entrena el **cascada binario + multiclase** (baseline integrado).
2. Entrena el estimador **`last_visible_idx`** y evalúa **clipping** sobre las predicciones de **ese mismo run**.

**No** requiere ejecutar otros notebooks `mejorafase*`. Tras adoptar una mejora en `train_spine_cascade_binary_to_thoracolumbar_v3.ipynb`, se **regenera** este par con `build_fase7_last_visible_notebooks.py`.

## Notebooks locales

| Variante | Archivo | Cascada (`OUTPUT_DIR`) | Last visible (`LAST_OUTPUT_DIR`) |
|----------|---------|------------------------|----------------------------------|
| CPU | `…_mejorafase7_auxiliares_rango_lastvis_cpu.ipynb` | `…/training_runs_cascade_v3_fase7_lastvis_cpu/` | `…/training_runs_last_visible_fase7_cpu/` |
| CUDA | `…_mejorafase7_auxiliares_rango_lastvis_cuda.ipynb` | `…/training_runs_cascade_v3_fase7_lastvis_cuda/` | `…/training_runs_last_visible_fase7_cuda/` |

Checkpoints en `models/` (generados al ejecutar **este** notebook):

- `binary_spine_cascade_fase7_lastvis_{cpu|cuda}_best.pt`
- `thoracolumbar_core_cascade_fase7_lastvis_{cpu|cuda}_best.pt`
- `last_visible_estimator_fase7_lastvis_{cpu|cuda}_best.pt`

## Regenerar

```text
python notebooks/v3_local/mejoras/scripts/build_fase7_last_visible_notebooks.py
```

Fuente cascada: `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3.ipynb`.  
Bloque last_visible: adaptado desde `notebooks/07_colab_train_last_visible_estimator_and_clip_thoracolumbar_explained.ipynb` (secciones 2–14), con inferencia alineada a `UNetSmall` del baseline.

## Flujo de cierre

1. Ejecutar `_cuda` en Colab (runtime completo: cascada + Fase 7).
2. Revisar CSV en `LAST_OUTPUT_DIR` y métricas cascada en `OUTPUT_DIR`.
3. Completar **Análisis e interpretación** en ese `.ipynb`.
4. Decisión en `../RESULTADOS_Y_DECISIONES_GENERAL.md` y `../experiment_registry.csv`.

**Anterior:** Fase 6 — `../train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase6_postproceso_ligero/README.md`.

**Siguiente:** Fase 8 — `../train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase8_eficiencia/README.md`.

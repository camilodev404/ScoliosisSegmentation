# Fase 8 — Eficiencia (`mejorafase8_eficiencia`)

**Estado:** **`_cuda` cerrado** (2026-05-15). **Decisión:** **No adoptar** (macro test **0,2157** vs Fase 4 **0,2380**; Δ **−0,0223** > umbral 0,01).

## Objetivo (plan §3 Fase 8)

Reducir coste computacional (menor resolución + U-Net compacta) con pérdida acotada de calidad:

- **Adoptar si:** `macro_dice_fg` test cae ≤ **0,01** absoluto vs Fase 4 `_cuda` (0,2380).
- **No adoptar** si la brecha supera ese presupuesto o hay regresión binaria relevante.

## Cambios experimentales (vs notebook vivo)

| Parámetro | Vivo (F3+F4) | Fase 8 `_cuda` | Fase 8 `_cpu` |
|-----------|--------------|----------------|---------------|
| `IMG_SIZE_*` | (512, 256) | **(384, 192)** | (256, 128) |
| `UNET_BASE_CHANNELS` | 32 (default) | **24** | **24** |
| Scheduler / augment | Sí | Igual | Igual |

## Notebooks

| Variante | Archivo | Salida métricas |
|----------|---------|-----------------|
| CUDA | `…_mejorafase8_eficiencia_cuda.ipynb` | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase8_eficiencia_cuda/` |
| CPU | `…_mejorafase8_eficiencia_cpu.ipynb` | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase8_eficiencia_cpu/` |

Checkpoints: `models/binary_spine_cascade_fase8_eficiencia_{cpu|cuda}_best.pt`, `models/thoracolumbar_core_cascade_fase8_eficiencia_{cpu|cuda}_best.pt`.

## Regenerar

```text
python notebooks/v3_local/mejoras/scripts/build_fase8_eficiencia_notebooks.py
```

Fuente: `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` (vivo con Fase 3 + 4).

## Flujo de cierre

1. Ejecutar `_cuda` en Colab.
2. Revisar CSV en `OUTPUT_DIR` y tiempos impresos al final de cada etapa.
3. Ejecutar `python notebooks/v3_local/mejoras/scripts/patch_fase8_eficiencia_analysis_cells.py` (tras el run).
4. Actualizar `../RESULTADOS_Y_DECISIONES_GENERAL.md` y `../experiment_registry.csv`.

**Anterior:** Fase 7 — `../train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase7_auxiliares_rango_lastvis/README.md`.

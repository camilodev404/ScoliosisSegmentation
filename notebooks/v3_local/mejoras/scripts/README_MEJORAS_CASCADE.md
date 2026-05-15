# Plantilla de notebooks de mejoras (cascada V3)

Los scripts `build_fase*_*.py` bajo esta carpeta generan variantes **`_cpu`** / **`_cuda`** desde
`notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3.ipynb`.

## Módulo común

`cascade_v3_mejora_notebook_common.py` concentra lo que se repite en casi todas las fases:

- Prefijo markdown (**mapa** del experimento) y, en `_cuda`, bloques **Comprobar CUDA** y **Montar Drive** (títulos markdown + dos celdas de código; IDE-safe fuera de Colab).
- Parche de la **celda de configuración**: `OUTPUT_DIR` bajo `outputs/analysis_outputs_v3/`, resolución, batch, épocas, print de depuración, ruta Colab “Other computers”.
- Sustitución de rutas de **checkpoints** binario / multiclase.
- Markdown **Antes de ejecutar** con la subcarpeta de salida.
- Aviso en **Cómo interpretar**, **registro de ejecución** al final (markdown + celda de código con fecha/hora al ejecutarla) y limpieza de `outputs` de celdas código.

Nueva fase que solo cambie hiperparámetros o código **después** de ROI: importar ese módulo, definir `VARIANTS` y aplicar **un** parche de texto sobre el notebook cargado (como Fase 2 y Fase 3).

## Alcance: ¿solo multiclase o pipeline más amplio?

| Tipo | Qué toca | Ejemplos en repo | Plantilla |
|------|-----------|------------------|-----------|
| **Solo multiclase** | Celdas de entrenamiento multiclase (pérdida, LR, scheduler, etc.); el binario y los helpers salvo config siguen el base. | Fase 2 (pesos CE T7–T12), Fase 3 (LR + cosine) | Mínimo: `VARIANTS` + parche + funciones comunes. |
| **Pre-multiclase / datos** | Helpers (`read_gray`, `prepare_*`), binario, splits o lógica antes del bucle multiclase. | Fase 1 (letterbox en crop multiclase) | Común **más** funciones específicas (p. ej. `patch_cell_helpers`). |

Si una mezcla requiere **binario distinto** o **ROI distinta**, conviene documentarlo explícitamente en el README de la fase y valorar si el parche sigue siendo un solo script o dos etapas (p. ej. regenerar desde un notebook intermedio).

## Convenciones

- Salidas bajo `outputs/analysis_outputs_v3/training_runs_cascade_v3_faseN_*_{cpu|cuda}/`.
- Registro: `experiment_registry.csv` y `RESULTADOS_Y_DECISIONES_GENERAL.md` (comparación global) **y**, en cada notebook ejecutado, celdas **Análisis e interpretación** con la lectura de **ese** run.

### Flujo de cierre por fase

1. `build_fase*_*.py` → par `_cpu` / `_cuda` con sección **Análisis e interpretación** **vacía** (no heredar tablas del notebook base).
2. Ejecutar (p. ej. `_cuda` en Colab) → CSV en `OUTPUT_DIR`.
3. Analizar → completar celdas del notebook **o** `python …/patch_faseN_*_analysis_cells.py` (solo ese `.ipynb` / semilla).
4. Decidir → `experiment_registry.csv` + `python notebooks/v3_local/mejoras/scripts/sync_registry_to_resultados.py` (actualiza la sección **Resumen final** en `RESULTADOS_Y_DECISIONES_GENERAL.md`); si se adopta, integrar en `train_spine_cascade_binary_to_thoracolumbar_v3.ipynb`.

**No** copiar tablas entre `_cpu` y `_cuda` ni entre semillas. Para reponer cierres `_cuda` ya documentados: `restore_executed_cuda_analysis_cells.py`.

## Fase 5 (multiseed)

`build_fase5_multiseed_notebooks.py` genera **tres** notebooks `_cuda` con distinta `SEED` y carpetas de salida (no hay par `_cpu`; el experimento es solo robustez estadística). **`patch_fase5_multiseed_analysis_cells.py`** solo debe ejecutarse **después** del run y **solo en el `.ipynb` de esa semilla** (no copiar el mismo bloque a las otras variantes).

## Fase 6 (post-proceso ligero)

`build_fase6_postproceso_notebooks.py` añade helpers y extiende `evaluate_multiclass` (flag `apply_fase6_postprocess`); el bloque final de test escribe métricas **sin post** y **con post** (`*_test_metrics_fase6_post.csv`, `*_per_class_metrics_fase6_post.csv`). Par `_cpu` / `_cuda` como en fases anteriores. **`patch_fase6_postproceso_analysis_cells.py`** (cifras ejemplo en el script) debe usarse **solo si** se desea documentar en el notebook esa misma ejecución; parámetros/editar el script para reflejar métricas **reales de esa variante**.

## Fase 7 (cascada + estimador last_visible / clipping)

`build_fase7_last_visible_notebooks.py` genera el par **autocontenido** desde `train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` (cascada completa) y anexa el bloque last_visible adaptado del notebook `07`. Salidas: `training_runs_cascade_v3_fase7_lastvis_{cpu|cuda}/` + `training_runs_last_visible_fase7_{cpu|cuda}/`. No depende de ejecutar otras `mejorafase*`. Ver README de `train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase7_auxiliares_rango_lastvis/`.

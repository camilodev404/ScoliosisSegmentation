# Resultados y decisiones — registro general (`mejoras/`)

Este documento **consolida** métricas, análisis y decisión de adopción por cada fase de mejora. El detalle (gráficas, tablas amplias, razonamiento extendido) debe vivir también en el **notebook o markdown del directorio de esa fase**; aquí se resume lo necesario para revisión rápida del historial.

**Documentación relacionada:** convenciones de carpetas y fases en `mejoras/PLAN_ACCION_AJUSTES_MODELOS.md` (§0–§8). **Plantillas copiables:** *Plantilla vacía — Ficha §7.1* y *Plantilla vacía — Tabla comparativa §7.2* más abajo.

**Normas:**

- No mover consumibles ni artefactos históricos de su ubicación original; solo referenciar rutas.
- Tras cada fase implementada o evaluada, añadir una **nueva sección** al final (o mantener el orden por número de fase).
- Incluir siempre: fecha, identificador de run/carpeta de salida, métricas clave, decisión (Adoptar / No adoptar / Adoptar con condición / Pendiente).
- Para cada fase o experimento, usar la **ficha §7.1** del plan: plantilla vacía más abajo (copiar) y, para la Fase 0, la ficha **rellenada** en la subsección correspondiente. Mantener además la **tabla §7.2** (plantilla vacía + tabla acumulativa en Fase 0) para comparativos tipo informe.

---

## Plantilla vacía — Ficha §7.1 (copiar por fase o experimento)

Definición de campos: `mejoras/PLAN_ACCION_AJUSTES_MODELOS.md` §7.1. Copiar este bloque al añadir **Fase 2, 3, …** o al documentar un experimento intermedio.

| Campo | Contenido |
|-------|-----------|
| **ID experimento** | *(rellenar)* |
| **Fase (número + nombre)** | *(rellenar)* |
| **Fecha cierre** | AAAA-MM-DD |
| **Responsable** | *(rellenar)* |
| **Hipótesis (1–2 frases)** | *(rellenar)* |
| **Cambio vs baseline** | *(rellenar)* |
| **Ruta directorio documentación** | `mejoras/..._mejorafaseN_.../` |
| **Ruta carpeta métricas generadas** | `analysis_outputs_v3/.../` |
| **Git commit** | *(rellenar)* |
| **Entorno** | *(Python / PyTorch / CUDA — rellenar)* |
| **Binario — test** | Dice: … \| IoU: … \| loss: … |
| **Multiclase — test** | `macro_dice_fg`: … \| `macro_iou_fg`: … \| loss: … \| `pixel_accuracy`: … (contexto) |
| **Mejor `val_macro_dice_fg` (época)** | *(rellenar)* |
| **Δ vs baseline (multiclase)** | *(rellenar; Fase 0: N/A)* |
| **Dice por clase (test)** | T9 …, T10 …, T11 …, L5 … *(ver `thoracolumbar_core_per_class_metrics.csv`)* |
| **ROI / calidad recorte** | *(rellenar; inspección / `thoracolumbar_core_binary_rois.csv`)* |
| **Estratificación (opcional)** | Normal: … \| Scoliosis: … |
| **Riesgos / efectos secundarios** | *(rellenar)* |
| **Conclusión (2–4 frases)** | *(rellenar)* |
| **Decisión** | **Adoptar** / **No adoptar** / **Adoptar con condición** |
| **Siguiente acción** | *(rellenar)* |

---

## Plantilla vacía — Tabla comparativa §7.2 (copiar para sección «Resultados»)

Definición: `mejoras/PLAN_ACCION_AJUSTES_MODELOS.md` §7.2. **Una fila por run o variante.** Copiar este bloque al preparar un cuadro comparativo o exportarlo a Excel.

| ID | Fase | `macro_dice_fg` (test) | `macro_iou_fg` (test) | Dice T9 | Dice T10 | Dice T11 | Dice L5 | Binario Dice (test) | Nota breve |
|----|------|------------------------|------------------------|---------|----------|----------|---------|----------------------|------------|
| *(id)* | *(N)* | … | … | … | … | … | … | … | *(rellenar)* |
| … | … | … | … | … | … | … | … | … | … |

---

## Fase 0 — Línea base y diagnóstico (`mejorafase0_base`)

### Ficha §7.1 — Fase 0 (baseline; completar campos pendientes)

| Campo | Contenido |
|-------|-----------|
| **ID experimento** | `fase0_baseline_cascade_v3_core` |
| **Fase (número + nombre)** | Fase 0 — Línea base y diagnóstico |
| **Fecha cierre** | *(completar al validar inspección ROI)* |
| **Responsable** | *(rellenar)* |
| **Hipótesis (1–2 frases)** | N/A: registro del estado de referencia del pipeline cascada V3 (subset core) antes de aplicar mejoras estructurales. |
| **Cambio vs baseline** | Ninguno (run de referencia ya ejecutado). |
| **Ruta directorio documentación** | `mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase0_base/` |
| **Ruta carpeta métricas generadas** | `analysis_outputs_v3/training_runs_cascade_v3/` |
| **Git commit** | *(rellenar al congelar baseline en repo)* |
| **Entorno** | *(rellenar si se audita reproducibilidad)* |
| **Binario — test** | Dice: **0,8713** \| IoU: **0,7720** \| loss: **0,2719** |
| **Multiclase — test** | `macro_dice_fg`: **0,1894** \| `macro_iou_fg`: **0,1126** \| loss: **2,3588** \| `pixel_accuracy`: **0,7487** (contexto ROI) |
| **Mejor `val_macro_dice_fg` (época)** | ≈ **0,1676** en época **24** (ver `thoracolumbar_core_history.csv`) |
| **Δ vs baseline (multiclase)** | N/A (esta fila **es** el baseline frente a fases ≥1). |
| **Dice por clase (test)** | Detalle en `analysis_outputs_v3/training_runs_cascade_v3/thoracolumbar_core_per_class_metrics.csv` (p. ej. L5 recuperada vs plano; revisar T9–T11). |
| **ROI / calidad recorte** | `thoracolumbar_core_binary_rois.csv`: 198/198 `pred_binary` en el análisis documentado; cualitativo: notebook `train_spine_cascade_binary_to_thoracolumbar_v3_inspection_ROI.ipynb` + `NOTAS_INSPECCION_ROI.md` (opcional). |
| **Estratificación (opcional)** | *(calcular y pegar si se estratifica por `split` Normal/Scoliosis)* |
| **Riesgos / efectos secundarios** | Métricas globales de píxel en ROI no comparables de forma ingenua con entrenamiento sobre imagen completa (plan §6 / informe 5.5). |
| **Conclusión (2–4 frases)** | Baseline cascada core registrado; macro Dice/IoU FG son las métricas principales de seguimiento. Completar conclusiones cualitativas de ROI tras inspección sistemática. |
| **Decisión** | **N/A (baseline)** — referencia para comparar fases ≥1. |
| **Siguiente acción** | Ejecutar / documentar inspección ROI; iniciar Fase 1 según `PLAN_ACCION_AJUSTES_MODELOS.md`. |

### Tabla comparativa §7.2 — inventario acumulado (actualizar al cerrar cada experimento)

Origen Dice por clase (fila baseline): `analysis_outputs_v3/training_runs_cascade_v3/thoracolumbar_core_per_class_metrics.csv` (test). Añadir **nuevas filas** debajo del baseline al registrar Fase 1, 2, …

| ID | Fase | `macro_dice_fg` (test) | `macro_iou_fg` (test) | Dice T9 | Dice T10 | Dice T11 | Dice L5 | Binario Dice (test) | Nota breve |
|----|------|------------------------|------------------------|---------|----------|----------|---------|----------------------|------------|
| `fase0_baseline_cascade_v3_core` | 0 | 0,1894 | 0,1126 | 0,0136 | 0,0440 | 0,0296 | 0,2158 | 0,8713 | Referencia `training_runs_cascade_v3`; T9–T11 desde CSV test |

Las subsecciones siguientes amplían las **mismas cifras** en tablas por archivo y el análisis narrativo.

### Métricas test (run cascada V3 en carpeta anterior; snapshot de referencia)

**Binario** (`binary_spine_test_metrics.csv`):

| loss | dice | iou | pixel_accuracy |
|------|------|-----|----------------|
| 0,2719 | 0,8713 | 0,7720 | 0,9612 |

**Multiclase core** (`thoracolumbar_core_test_metrics.csv`):

| loss | pixel_accuracy | macro_dice_fg | macro_iou_fg |
|------|----------------|---------------|--------------|
| 2,3588 | 0,7487 | 0,1894 | 0,1126 |

### Análisis (resumen)

- La cascada mejora frente al pipeline plano en `macro_dice_fg` / `macro_iou_fg`; la `pixel_accuracy` multiclase es menor que en imagen completa por menor fondo en ROI (esperado).
- Hallazgos por clase (detalle en `thoracolumbar_core_per_class_metrics.csv` y en el bloque de análisis del notebook cascada): recuperación de **L5** frente al plano; atención a **T9–T11** en cascada.
- ROI: revisar `thoracolumbar_core_binary_rois.csv` y el notebook de inspección para patrones de recorte.

### Decisión

Ver fila **Decisión** y **Conclusión** en la **Ficha §7.1** arriba (baseline: **N/A**).

### Próxima acción sugerida

Documentar en `mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase0_base/` (p. ej. `NOTAS_INSPECCION_ROI.md`) los hallazgos visuales tras ejecutar el notebook de inspección.

---

## Fase 1 — Letterbox ROI multiclase (`mejorafase1_letterbox_roi`)

**Estado:** notebooks `_cpu` y `_cuda` generados; **pendiente de ejecutar** en Jupyter según hardware.

| Campo | Valor |
|-------|--------|
| **Directorio de la fase** | `mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_letterbox_roi/` |
| **Notebooks** | `…/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_letterbox_roi_cpu.ipynb` (PC sin GPU viable) y `…_cuda.ipynb` (referencia comparable al baseline) |
| **Regenerar** | `python mejoras/scripts/build_fase1_letterbox_notebooks.py` (si cambia el cascada V3 base) |
| **Salida de métricas** | `analysis_outputs_v3/training_runs_cascade_v3_fase1_letterbox_cpu/` y `…_fase1_letterbox_cuda/` |
| **Pesos** | `*_fase1_letterbox_cpu_best.pt` y `*_fase1_letterbox_cuda_best.pt` bajo `models/` (ver README de la fase) |
| **Fecha** | *(al cerrar el run)* |
| **Hipótesis** | Letterbox reduce distorsión anisotrópica del crop → mejor separación entre vértebras adyacentes (p. ej. T9–T11). |

Tras ejecutar: copiar **Plantilla §7.1** arriba, añadir fila en **Tabla §7.2** (Fase 0), completar subsecciones siguientes.

### Métricas (vs Fase 0)

*(rellenar: baseline \| fase1 \| Δ — desde los CSV de ambas carpetas)*

### Análisis

*(rellenar)*

### Decisión

*(rellenar — criterios del plan §1.2 y §4.2)*

---

*Ir añadiendo secciones Fase 2, Fase 3, … cada una con una **Ficha §7.1** (copiar la plantilla vacía de arriba) y una fila nueva en la **Tabla comparativa §7.2** (Fase 0 o sección global).*

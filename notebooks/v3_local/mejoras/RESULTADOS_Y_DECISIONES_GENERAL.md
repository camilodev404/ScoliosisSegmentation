# Resultados y decisiones — registro general (`notebooks/v3_local/mejoras/`)

Este documento **consolida** métricas, análisis y decisión de adopción por cada fase de mejora en el repositorio **`ScoliosisSegmentation`**. El detalle (gráficas, tablas amplias, razonamiento extendido) debe vivir también en el **notebook o markdown del directorio de esa fase**; aquí se resume lo necesario para revisión rápida del historial.

**Rutas canónicas (raíz del repo):** plan y registro en `notebooks/v3_local/mejoras/`; métricas y manifiestos en `outputs/analysis_outputs_v3/`; dataset en `data/Scoliosis_Dataset/`; pesos en `models/`.

**Esquema dual (obligatorio para fases con entrenamiento):** registrar por separado runs **`_cpu`** (Jupyter local, perfil liviano) y **`_cuda`** (Google Colab u otra GPU; métricas de referencia para comparar con el baseline). En Colab usar `MAIA_PROJECT_ROOT` o `%cd` a la raíz del clon antes de ejecutar el notebook `*_cuda.ipynb`.

**Documentación relacionada:** convenciones en `notebooks/v3_local/mejoras/PLAN_ACCION_AJUSTES_MODELOS.md` (§0–§8). **Registro tabular (fuente para el resumen):** `notebooks/v3_local/mejoras/experiment_registry.csv`. **Sincronizar resumen final:** `python notebooks/v3_local/mejoras/scripts/sync_registry_to_resultados.py` → sección [Resumen final — registro de experimentos](#resumen-final--registro-de-experimentos). **Plantillas copiables:** *Plantilla vacía — Ficha §7.1* y *Plantilla vacía — Tabla comparativa §7.2* más abajo.

**Cierre del plan — notebook final:** el procedimiento vigente de entrenamiento es `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` (Fase 3 + Fase 4 integradas). La **mejor corrida adoptada** documentada por métricas globales es el run **`4_cuda`** (Fase 4; `macro_dice_fg` test 0,2380); el notebook de experimento asociado y las rutas de salida están en `notebooks/v3_local/mejoras/README.md` § *Notebook final y mejor run de referencia*. Un run puntual (`5_cuda_s1337`) supera el macro de `4_cuda` pero no se consolidó; no sustituye al pipeline aprobado.

**Normas:**

- No mover consumibles ni artefactos históricos de su ubicación original; solo referenciar rutas.
- Tras cada fase implementada o evaluada, añadir una **nueva sección** al final (o mantener el orden por número de fase).
- Incluir siempre: fecha, identificador de run/carpeta de salida, métricas clave, decisión (Adoptar / No adoptar / Adoptar con condición / Pendiente).
- Para cada fase o experimento, usar la **ficha §7.1** del plan: plantilla vacía más abajo (copiar) y, para la Fase 0, la ficha **rellenada** en la subsección correspondiente. Mantener además la **tabla §7.2** (plantilla vacía + tabla acumulativa en Fase 0) para comparativos tipo informe.

---

## Plantilla vacía — Ficha §7.1 (copiar por fase o experimento)

Definición de campos: `notebooks/v3_local/mejoras/PLAN_ACCION_AJUSTES_MODELOS.md` §7.1. Copiar este bloque al añadir **Fase 2, 3, …** o al documentar un experimento intermedio.

| Campo | Contenido |
|-------|-----------|
| **ID experimento** | *(rellenar)* |
| **Fase (número + nombre)** | *(rellenar)* |
| **Fecha cierre** | AAAA-MM-DD |
| **Responsable** | *(rellenar)* |
| **Hipótesis (1–2 frases)** | *(rellenar)* |
| **Cambio vs baseline** | *(rellenar)* |
| **Ruta directorio documentación** | `notebooks/v3_local/mejoras/..._mejorafaseN_.../` |
| **Ruta carpeta métricas generadas** | `outputs/analysis_outputs_v3/.../` |
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

Definición: `notebooks/v3_local/mejoras/PLAN_ACCION_AJUSTES_MODELOS.md` §7.2. **Una fila por run o variante.** Copiar este bloque al preparar un cuadro comparativo o exportarlo a Excel.

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
| **Ruta directorio documentación** | `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase0_base/` |
| **Ruta carpeta métricas generadas** | `outputs/analysis_outputs_v3/training_runs_cascade_v3/` |
| **Git commit** | *(rellenar al congelar baseline en repo)* |
| **Entorno** | *(rellenar si se audita reproducibilidad)* |
| **Binario — test** | Dice: **0,8713** \| IoU: **0,7720** \| loss: **0,2719** |
| **Multiclase — test** | `macro_dice_fg`: **0,1894** \| `macro_iou_fg`: **0,1126** \| loss: **2,3588** \| `pixel_accuracy`: **0,7487** (contexto ROI) |
| **Mejor `val_macro_dice_fg` (época)** | ≈ **0,1676** en época **24** (ver `thoracolumbar_core_history.csv`) |
| **Δ vs baseline (multiclase)** | N/A (esta fila **es** el baseline frente a fases ≥1). |
| **Dice por clase (test)** | Detalle en `outputs/analysis_outputs_v3/training_runs_cascade_v3/thoracolumbar_core_per_class_metrics.csv` (p. ej. L5 recuperada vs plano; revisar T9–T11). |
| **ROI / calidad recorte** | `thoracolumbar_core_binary_rois.csv`: 198/198 `pred_binary` en el análisis documentado; cualitativo: `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3_inspection_ROI.ipynb` + `NOTAS_INSPECCION_ROI.md` (opcional). |
| **Estratificación (opcional)** | *(calcular y pegar si se estratifica por `split` Normal/Scoliosis)* |
| **Riesgos / efectos secundarios** | Métricas globales de píxel en ROI no comparables de forma ingenua con entrenamiento sobre imagen completa (plan §6 / informe 5.5). |
| **Conclusión (2–4 frases)** | Baseline cascada core registrado; macro Dice/IoU FG son las métricas principales de seguimiento. Completar conclusiones cualitativas de ROI tras inspección sistemática. |
| **Decisión** | **N/A (baseline)** — referencia para comparar fases ≥1. |
| **Siguiente acción** | **Fases 5–8** documentadas (Fase 8: **No adoptar** eficiencia 384×192/base=24). Cierre del plan de mejoras cascada según `PLAN_ACCION_AJUSTES_MODELOS.md`. |

### Tabla comparativa §7.2 — inventario acumulado (actualizar al cerrar cada experimento)

Origen Dice por clase (fila baseline): `outputs/analysis_outputs_v3/training_runs_cascade_v3/thoracolumbar_core_per_class_metrics.csv` (test). Añadir **nuevas filas** debajo del baseline al registrar Fase 1, 2, …

| ID | Fase | `macro_dice_fg` (test) | `macro_iou_fg` (test) | Dice T9 | Dice T10 | Dice T11 | Dice L5 | Binario Dice (test) | Nota breve |
|----|------|------------------------|------------------------|---------|----------|----------|---------|----------------------|------------|
| `fase0_baseline_cascade_v3_core` | 0 | 0,1894 | 0,1126 | 0,0136 | 0,0440 | 0,0296 | 0,2158 | 0,8713 | Referencia `training_runs_cascade_v3`; T9–T11 desde CSV test |
| `fase1_letterbox_cuda_2026-05-14` | 1 | 0,1807 | 0,1038 | 0,1035 | 0,0279 | 0,1079 | 0,2330 | 0,8581 | `training_runs_cascade_v3_fase1_letterbox_cuda`; letterbox multiclase; **No adoptar** (ver Fase 1) |
| `fase2_pesos_t7_t12_cuda_2026-05-14` | 2 | 0,1896 | 0,1104 | 0,0592 | 0,1144 | 0,0566 | 0,1470 | 0,8767 | `training_runs_cascade_v3_fase2_pesos_t7_t12_cuda`; CE ×1,40 en T7–T12; **No adoptar** (ver Fase 2) |
| `fase3_scheduler_lr_cuda_2026-05-15` | 3 | 0,2323 | 0,1375 | 0,1354 | 0,0644 | 0,1453 | 0,2025 | 0,8611 | Run A `…_fase3_scheduler_lr_cuda`; LR×0,5 + CosineAnnealingLR; **Adoptar con condición** (histórico; ver run B) |
| `fase3_scheduler_lr_cuda_repro_b_2026-05-14` | 3 | **0,2345** | **0,1387** | 0,1734 | 0,0274 | 0,1430 | 0,2103 | **0,8710** | Run B `…_fase3_scheduler_lr_cuda_repro_b`; misma lógica + cudnn deterministic; confirma binario ~baseline; **Adoptar** (ver Fase 3) |
| `fase4_augment_roi_cuda_2026-05-14` | 4 | **0,2380** | **0,1415** | 0,1324 | **0,0731** | **0,1750** | 0,1955 | **0,8710** | `…_fase4_augment_roi_cuda`; augment train ROI; macro **+0,0035** vs run B; **Adoptar** (ver Fase 4) |
| `fase5_multiseed_cuda_s42_2026-05-15` | 5 | 0,2294 | 0,1365 | 0,0935 | 0,0831 | 0,1977 | 0,1807 | 0,8710 | `…_fase5_multiseed_cuda_s42`; misma pipeline vivo; **Prometedor no consolidado** (ver Fase 5) |
| `fase5_multiseed_cuda_s1337_2026-05-15` | 5 | **0,2457** | **0,1503** | 0,1480 | 0,1261 | 0,0800 | **0,4178** | **0,8943** | `…_fase5_multiseed_cuda_s1337`; mejor macro de la tanda vs Fase 4 |
| `fase5_multiseed_cuda_s4242_2026-05-15` | 5 | 0,1737 | 0,1035 | 0,1283 | **0,0000** | 0,1653 | 0,3326 | 0,8582 | `…_fase5_multiseed_cuda_s4242`; macro por debajo del baseline; T10 colapsada |
| `fase6_postproc_cuda_2026-05-15` | 6 | **0,2294** | **0,1360** | 0,1128 | 0,0777 | 0,1783 | 0,1925 | 0,8710 | `…_fase6_postproc_cuda`; métricas **sin post** (con post islas K=0: macro **0,2273**); **No adoptar** (ver Fase 6) |
| `fase7_lastvis_cuda_2026-05-15` | 7 | **0,2258** | **0,1341** | 0,0872 | 0,0849 | 0,1771 | 0,1871 | 0,8710 | `…_fase7_lastvis_cuda` + `…_last_visible_fase7_cuda`; clip +0,0079 vs raw; **No adoptar** (ver Fase 7) |
| `fase8_eficiencia_cuda_2026-05-15` | 8 | **0,2157** | **0,1319** | 0,1946 | 0,1100 | 0,1618 | 0,2785 | 0,8906 | `…_fase8_eficiencia_cuda`; 384×192 base=24; macro −0,0223 vs F4; **No adoptar** (ver Fase 8) |

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

Documentar en `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase0_base/` (p. ej. `NOTAS_INSPECCION_ROI.md`) los hallazgos visuales tras ejecutar el notebook de inspección.

---

## Fase 1 — Letterbox ROI multiclase (`mejorafase1_letterbox_roi`)

**Estado:** variante **`_cuda`** ejecutada y registrada (2026-05-14). Variante **`_cpu`** sigue siendo opcional para iteración local; la decisión de fase se basa en **`_cuda`** frente al baseline.

| Campo | Valor |
|-------|--------|
| **Directorio de la fase** | `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_letterbox_roi/` |
| **Notebooks** | `…_letterbox_roi_cpu.ipynb`, `…_letterbox_roi_cuda.ipynb` |
| **Regenerar** | `python notebooks/v3_local/mejoras/scripts/build_fase1_letterbox_notebooks.py` |
| **Salida métricas `_cuda`** | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase1_letterbox_cuda/` |
| **Salida métricas `_cpu`** | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase1_letterbox_cpu/` (si se ejecuta) |
| **Análisis en notebook** | `train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_letterbox_roi_cuda.ipynb` (celdas finales actualizadas) |

### Ficha §7.1 — Fase 1, experimento `1_cuda` (letterbox, referencia Colab)

| Campo | Contenido |
|-------|-----------|
| **ID experimento** | `fase1_letterbox_cuda_2026-05-14` |
| **Fase (número + nombre)** | Fase 1 — Letterbox en ROI multiclase (`_cuda`) |
| **Fecha cierre** | 2026-05-14 |
| **Responsable** | *(rellenar)* |
| **Hipótesis (1–2 frases)** | El letterbox en el crop multiclase (escala uniforme + padding) reduce distorsión frente al resize directo y mejora o mantiene `macro_dice_fg` en test frente al cascada V3 base. |
| **Cambio vs baseline** | Sustitución del resize anisotrópico del crop multiclase por `letterbox_gray_and_mask`; mismo flujo binario → ROI → multiclase y mismas épocas/resolución que el perfil completo del base. |
| **Ruta directorio documentación** | `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_letterbox_roi/` |
| **Ruta carpeta métricas generadas** | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase1_letterbox_cuda/` |
| **Git commit** | *(rellenar al congelar en repo)* |
| **Entorno** | Colab + GPU (CUDA); PyTorch según runtime Colab |
| **Binario — test** | Dice: **0,8581** \| IoU: **0,7514** \| loss: **0,3166** |
| **Multiclase — test** | `macro_dice_fg`: **0,1807** \| `macro_iou_fg`: **0,1038** \| loss: **2,1910** \| `pixel_accuracy`: **0,7754** (contexto) |
| **Mejor `val_macro_dice_fg` (época)** | ≈ **0,2123** en época **24** (`thoracolumbar_core_history.csv` del run) |
| **Δ vs baseline (multiclase)** | `macro_dice_fg`: **−0,0087** \| `macro_iou_fg`: **−0,0088** (baseline: 0,1894 / 0,1126 en `training_runs_cascade_v3/`) |
| **Dice por clase (test)** | T9 **0,1035**, T10 **0,0279**, T11 **0,1079**, L5 **0,2330** (T10 peor que baseline ~0,044; T9/T11 mejoran). Detalle: `thoracolumbar_core_per_class_metrics.csv`. |
| **ROI / calidad recorte** | `thoracolumbar_core_binary_rois.csv`: **198/198** `pred_binary`, sin `fallback_full_image`. |
| **Estratificación (opcional)** | *(pendiente)* |
| **Riesgos / efectos secundarios** | Mejor `val_macro_dice_fg` que el baseline pero **test** macro menor: posible variabilidad de split/regularización; ligera baja del binario en test. |
| **Conclusión (2–4 frases)** | Con el criterio del plan (§1.2 / §4.2), el letterbox **no** mejora las métricas principales de test frente al cascada base: bajan `macro_dice_fg` e `macro_iou_fg`. Hay reparto mixto por vértebra (T9/T11 suben, T10 y parte alta bajan). El ROI sigue siendo estable. |
| **Decisión** | **No adoptar** integrar letterbox en el procedimiento base con esta configuración. |
| **Siguiente acción** | Tras cierre Fase 1 (letterbox): avanzar a **Fase 2** (`…_mejorafase2_pesos_clases_t7_t12/`) según el plan. |

### Métricas (test): baseline cascada vs Fase 1 `_cuda`

| Métrica | Baseline `training_runs_cascade_v3` | Fase 1 `_cuda` `…_fase1_letterbox_cuda` |
|--------|-------------------------------------|----------------------------------------|
| Binario — Dice | 0,8713 | 0,8581 |
| Binario — IoU | 0,7720 | 0,7514 |
| Multiclase — `macro_dice_fg` | 0,1894 | 0,1807 |
| Multiclase — `macro_iou_fg` | 0,1126 | 0,1038 |

### Análisis (resumen)

- **Decisión principal:** según `macro_dice_fg` / `macro_iou_fg` en test, **no** hay mejora frente al baseline; la hipótesis de letterbox **no** queda apoyada para adopción.
- **Validación:** `val_macro_dice_fg` máximo (~0,212, época 24) **supera** al del baseline (~0,168); conviene no sobreinterpretar test único sin repetición por semilla si se quisiera explorar más el letterbox.
- **ROI:** estable; el problema no parece ser fallo masivo de ROI sino el efecto del cambio geométrico + entrenamiento en el multiclase.
- **Detalle:** ver tabla §7.2 arriba y notebook `_cuda` (sección de análisis).

### Decisión (cierre Fase 1)

**No adoptar** el letterbox multiclase como parte del procedimiento base **en este experimento**. Avanzar a **Fase 2** según plan.

---

## Fase 2 — Pérdida / muestreo T7–T12 (`mejorafase2_pesos_clases_t7_t12`)

**Estado:** variante **`_cuda`** ejecutada y registrada (2026-05-14). Variante **`_cpu`** sigue siendo opcional para iteración local; la decisión de fase se basa en **`_cuda`** frente al baseline.

| Campo | Valor |
|-------|--------|
| **Directorio de la fase** | `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase2_pesos_clases_t7_t12/` |
| **Notebooks** | `…_pesos_clases_t7_t12_cpu.ipynb`, `…_pesos_clases_t7_t12_cuda.ipynb` |
| **Regenerar** | `python notebooks/v3_local/mejoras/scripts/build_fase2_pesos_t7_t12_notebooks.py` |
| **Salida métricas `_cuda`** | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase2_pesos_t7_t12_cuda/` |
| **Salida métricas `_cpu`** | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase2_pesos_t7_t12_cpu/` (si se ejecuta) |
| **Análisis en notebook** | `train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase2_pesos_clases_t7_t12_cuda.ipynb` (celdas finales actualizadas) |

### Ficha §7.1 — Fase 2, experimento `2_cuda` (pesos CE T7–T12, referencia Colab)

| Campo | Contenido |
|-------|-----------|
| **ID experimento** | `fase2_pesos_t7_t12_cuda_2026-05-14` |
| **Fase (número + nombre)** | Fase 2 — Refuerzo CE en T7–T12 (`_cuda`, multiplicador **1,40**) |
| **Fecha cierre** | 2026-05-14 |
| **Responsable** | *(rellenar)* |
| **Hipótesis (1–2 frases)** | Aumentar el peso de la CE en **T7–T12** mejora el Dice en esa banda torácica media-inferior sin empeorar de forma relevante el **macro** de foreground ni clases críticas como **L5**. |
| **Cambio vs baseline** | Tras `estimate_multiclass_class_weights`, se multiplica el vector CE de `class_id` **7–12** por **1,40** (tensor y `class_weights_df`); misma arquitectura, épocas y letterbox que el cascada V3 base (sin Fase 1). |
| **Ruta directorio documentación** | `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase2_pesos_clases_t7_t12/` |
| **Ruta carpeta métricas generadas** | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase2_pesos_t7_t12_cuda/` |
| **Git commit** | *(rellenar al congelar en repo)* |
| **Entorno** | Colab + GPU (CUDA); PyTorch según runtime Colab |
| **Binario — test** | Dice: **0,8767** \| IoU: **0,7805** \| loss: **0,2700** (aprox. desde CSV) |
| **Multiclase — test** | `macro_dice_fg`: **0,1896** \| `macro_iou_fg`: **0,1104** \| loss: **2,4373** \| `pixel_accuracy`: **0,7221** |
| **Mejor `val_macro_dice_fg` (época)** | ≈ **0,2019** en época **21** (`thoracolumbar_core_history.csv` del run) |
| **Δ vs baseline (multiclase)** | `macro_dice_fg`: **+0,0002** (0,1896 vs 0,1894) \| `macro_iou_fg`: **−0,0022** (0,1104 vs 0,1126) en `training_runs_cascade_v3/` |
| **Dice por clase (test)** | **T7–T12** suben claramente vs baseline (p. ej. T7 **0,161** vs **0,074**; T10 **0,114** vs **0,044**; T12 **0,128** vs **0,091**). **L5** baja (**0,147** vs **0,216**). Detalle: `thoracolumbar_core_per_class_metrics.csv`. |
| **ROI / calidad recorte** | `thoracolumbar_core_binary_rois.csv`: **198/198** `pred_binary`, sin `fallback_full_image`. |
| **Estratificación (opcional)** | *(pendiente)* |
| **Riesgos / efectos secundarios** | Reponderar solo la **CE** desalinea el equilibrio CE+Dice: mejora local en T7–T12 con **coste** en L5 y ligera baja de `macro_iou_fg`; `pixel_accuracy` multiclase también baja vs baseline. |
| **Conclusión (2–4 frases)** | La hipótesis de ayudar T7–T12 **se cumple** en Dice por clase, pero el criterio global del plan (`macro_dice_fg` + `macro_iou_fg`) **no** mejora de forma clara: el macro Dice apenas cambia y el macro IoU empeora levemente. La regresión de **L5** es un contraindicador fuerte frente al logro del baseline cascada. |
| **Decisión** | **No adoptar** integrar este multiplicador **1,40×** solo en CE como procedimiento base. |
| **Siguiente acción** | Explorar variantes acotadas: multiplicador más bajo (p. ej. 1,15–1,25), peso análogo en la **rama Dice**, o focal suave; documentar en Fase 3 / experimentos según `PLAN_ACCION_AJUSTES_MODELOS.md`. |

### Métricas (test): baseline cascada vs Fase 2 `_cuda`

| Métrica | Baseline `training_runs_cascade_v3` | Fase 2 `_cuda` `…_fase2_pesos_t7_t12_cuda` |
|--------|-------------------------------------|-------------------------------------------|
| Binario — Dice | 0,8713 | 0,8767 |
| Binario — IoU | 0,7720 | 0,7805 |
| Multiclase — `macro_dice_fg` | 0,1894 | 0,1896 |
| Multiclase — `macro_iou_fg` | 0,1126 | 0,1104 |

### Análisis (resumen)

- **Decisión principal:** con el criterio acordado (macro de foreground + salud por clase clave), **no** se adopta el cambio tal cual: **macro IoU** baja, **L5** cae respecto al baseline, y la subida de **macro Dice** es marginal (+0,0002).
- **Hallazgo útil:** el refuerzo en CE **sí** desplaza el modelo hacia mejor Dice en **T7–T12** en test; sirve como pista para experimentos más conservadores o para combinar con la rama Dice.
- **ROI:** estable (198/198 `pred_binary`); el efecto observado no se atribuye a fallos masivos de ROI.
- **Validación:** el mejor `val_macro_dice_fg` (~0,202, época 21) supera al del baseline en validación (~0,168); el test macro sigue casi plano frente al baseline — conviene no sobreinterpretar sin repetición por semilla.

### Decisión (cierre Fase 2)

**No adoptar** el multiplicador **1,40×** en CE solo sobre T7–T12 como parte del procedimiento base **en este experimento**. Avanzar a iteraciones más suaves o a otras palancas del plan (Fase 3 / experimentos).

---

## Fase 3 — Optimización multiclase: LR + scheduler (`mejorafase3_scheduler_lr`)

**Estado:** variante **`_cuda`** (run A) y **verificación run B** (`…_cuda_repro_verificacion.ipynb`) ejecutadas y registradas. Variante **`_cpu`** opcional para iteración local.

| Campo | Valor |
|-------|--------|
| **Directorio de la fase** | `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase3_scheduler_lr/` |
| **Notebooks** | `…_scheduler_lr_cpu.ipynb`, `…_scheduler_lr_cuda.ipynb`, `…_scheduler_lr_cuda_repro_verificacion.ipynb` |
| **Regenerar** | `python notebooks/v3_local/mejoras/scripts/build_fase3_scheduler_lr_notebooks.py` · verificación: `python notebooks/v3_local/mejoras/scripts/build_fase3_scheduler_lr_cuda_repro_notebook.py` |
| **Salida métricas `_cuda` (run A)** | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase3_scheduler_lr_cuda/` |
| **Salida métricas verificación (run B)** | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase3_scheduler_lr_cuda_repro_b/` |
| **Análisis en notebook** | `…_scheduler_lr_cuda.ipynb` y `…_scheduler_lr_cuda_repro_verificacion.ipynb` (celdas finales actualizadas) |

### Ficha §7.1 — Fase 3, experimento `3_cuda` (LR multiclase ×0,5 + CosineAnnealingLR)

| Campo | Contenido |
|-------|-----------|
| **ID experimento** | `fase3_scheduler_lr_cuda_2026-05-15` |
| **Fase (número + nombre)** | Fase 3 — Optimización multiclase (scheduler + LR reducido, `_cuda`) |
| **Fecha cierre** | 2026-05-15 |
| **Responsable** | *(rellenar)* |
| **Hipótesis (1–2 frases)** | Reducir LR del multiclase y usar cosine annealing estabiliza el entrenamiento sobre ROI y mejora `macro_dice_fg` / `macro_iou_fg` en test frente al cascada base sin dañar de forma grave L5 ni el binario. |
| **Cambio vs baseline** | `LR_MULTICLASS = LR × 0,5`, `CosineAnnealingLR(T_max=MULTICLASS_EPOCHS, eta_min≈1% LR multiclase)` y `scheduler.step()` por época; binario sin cambios explícitos en hiperparámetros. |
| **Ruta directorio documentación** | `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase3_scheduler_lr/` |
| **Ruta carpeta métricas generadas** | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase3_scheduler_lr_cuda/` |
| **Git commit** | *(rellenar al congelar en repo)* |
| **Entorno** | Colab + GPU (CUDA); PyTorch según runtime Colab |
| **Binario — test** | Dice: **0,8611** \| IoU: **0,7560** \| loss: **0,3020** (aprox.) |
| **Multiclase — test** | `macro_dice_fg`: **0,2323** \| `macro_iou_fg`: **0,1375** \| loss: **2,2381** \| `pixel_accuracy`: **0,7519** |
| **Mejor `val_macro_dice_fg` (época)** | ≈ **0,2345** en época **22** (`thoracolumbar_core_history.csv`) |
| **Δ vs baseline (multiclase)** | `macro_dice_fg`: **+0,0429** \| `macro_iou_fg`: **+0,0249** (baseline: 0,1894 / 0,1126) |
| **Dice por clase (test)** | T9 **0,1354**, T10 **0,0644**, T11 **0,1453**, L5 **0,2025** (T9–T11 muy por encima del baseline; L5 levemente por debajo de **0,2158**). Detalle: `thoracolumbar_core_per_class_metrics.csv`. |
| **ROI / calidad recorte** | `thoracolumbar_core_binary_rois.csv`: **198/198** `pred_binary`. |
| **Estratificación (opcional)** | *(pendiente)* |
| **Riesgos / efectos secundarios** | En el run A, el binario test quedó ~0,01 por debajo del baseline sin cambio de LR en código; la **verificación run B** (misma lógica + `cudnn` más determinista) mostró que **no** es una regresión sistemática (ver ficha siguiente). |
| **Conclusión (2–4 frases)** | Run A: el patrón Fase 3 cumple el criterio principal en **macro** multiclase; T9–T11 suben fuerte; L5 baja leve vs baseline. La bajada del binario en A motivó la condición de adopción. |
| **Decisión** | **Adoptar con condición** (cierre run A, 2026-05-15) — pendiente verificación de reproducibilidad del binario. |
| **Siguiente acción** | *(Cumplida)* Ejecutar y registrar verificación `3_cuda_repro_b` (ver ficha siguiente). |

### Ficha §7.1 — Fase 3, verificación `3_cuda_repro_b` (repetición run B)

| Campo | Contenido |
|-------|-----------|
| **ID experimento** | `fase3_scheduler_lr_cuda_repro_b_2026-05-14` |
| **Fase (número + nombre)** | Fase 3 — Repetición `_cuda` (run B, carpeta `…_cuda_repro_b`) |
| **Fecha cierre** | 2026-05-14 |
| **Responsable** | *(rellenar)* |
| **Hipótesis (1–2 frases)** | Repetir el entrenamiento del run A con otra carpeta de salida y flags `cudnn` más deterministas; si el binario en test vuelve al nivel del baseline y el macro multiclase se mantiene alto, se levanta la condición de adopción. |
| **Cambio vs run A** | Misma arquitectura, épocas, `LR_MULTICLASS`, scheduler y `SEED=42`; distintos `OUTPUT_DIR` / checkpoints; tras `torch.cuda.manual_seed_all(SEED)` se fija `cudnn.benchmark=False` y `cudnn.deterministic=True`. |
| **Ruta carpeta métricas** | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase3_scheduler_lr_cuda_repro_b/` |
| **Binario — test** | Dice: **0,8710** \| IoU: **0,7714** \| loss: **0,2812** |
| **Multiclase — test** | `macro_dice_fg`: **0,2345** \| `macro_iou_fg`: **0,1387** \| loss: **2,2232** \| `pixel_accuracy`: **0,7516** |
| **Mejor `val_macro_dice_fg` (época)** | ≈ **0,2214** en época **22** (puede diferir de A por cudnn; criterio de adopción basado en **test** + binario). |
| **Δ vs baseline (multiclase)** | `macro_dice_fg`: **+0,0451** \| `macro_iou_fg`: **+0,0261** (baseline: 0,1894 / 0,1126). |
| **Dice por clase (test)** | T9 **0,1734**, T10 **0,0274**, T11 **0,1430**, L5 **0,2103** (vs A: T9/L5 mejor; T10 peor). |
| **ROI** | **198/198** `pred_binary`. |
| **Conclusión (2–4 frases)** | Run B alinea el **binario** con el baseline (Dice/IoU) y **mejora** ligeramente el macro multiclase respecto al run A. La caída del binario observada solo en A se interpreta como **variación por no-determinismo** en GPU, no como efecto del scheduler multiclase. |
| **Decisión** | **Adoptar** (sin condición pendiente sobre el binario) el esquema **LR multiclase ×0,5 + `CosineAnnealingLR`** como referencia para cascada core. **Integración en repo:** misma lógica fusionada en `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` (incl. `cudnn` reproducible tras semilla). Siguiente paso operativo: **Fase 4** y/o espejo en Colab `03_…` si se requiere. |
| **Siguiente acción** | **Fase 4** (augmentación ROI) y/o integración al notebook base; documentar en `experiment_registry.csv` la fila `3_cuda_repro_b`. |

### Métricas (test): baseline vs run A vs run B

| Métrica | Baseline | Run A | Run B |
|--------|----------|-------|-------|
| Binario — Dice | 0,8713 | 0,8611 | **0,8710** |
| Binario — IoU | 0,7720 | 0,7560 | **0,7714** |
| Multiclase — `macro_dice_fg` | 0,1894 | 0,2323 | **0,2345** |
| Multiclase — `macro_iou_fg` | 0,1126 | 0,1375 | **0,1387** |

### Análisis (resumen)

- **Macro multiclase:** run A y run B **superan** con claridad al baseline; run B **+0,0022** en `macro_dice_fg` vs A.
- **Binario:** run B **recupera** el nivel del baseline; la lectura del run A aislado **sobreinterpretaba** una regresión binaria inexistente en condiciones más deterministas.
- **Por clase:** trade-off T9/L5 vs T10 entre A y B; conviene seguir monitoreando **T10** en Fase 4.
- **ROI:** estable en ambos runs (198/198).

### Decisión (cierre Fase 3, incluye verificación)

**Adoptar** el esquema **LR multiclase ×0,5 + `CosineAnnealingLR`** como referencia para entrenamientos cascada core, con la evidencia del **run B** que levanta la condición planteada tras el run A. La lógica adoptada quedó **integrada** en `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3.ipynb`. **Fase 4** (augment ROI train) y **Fase 5** (multiseed) quedaron **cerradas**; Fase 5 documenta varianza por semilla sin cambiar el vivo (ver sección Fase 5). Siguiente paso natural: **Fase 6** (post-proceso ligero) y/o Colab `03_…`.

---

## Fase 4 — Augmentación geométrica suave en ROI multiclase (`mejorafase4_augmentacion_roi`)

**Estado:** variante **`_cuda`** ejecutada y cerrada (2026-05-14). Variante **`_cpu`** opcional. **Decisión:** **Adoptar** augment ROI en train; integrada en el notebook vivo.

| Campo | Valor |
|-------|--------|
| **Directorio de la fase** | `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase4_augmentacion_roi/` |
| **Notebooks** | `…_augmentacion_roi_cpu.ipynb`, `…_augmentacion_roi_cuda.ipynb` |
| **Regenerar** | `python notebooks/v3_local/mejoras/scripts/build_fase4_augment_roi_notebooks.py` |
| **Salida métricas `_cuda`** | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase4_augment_roi_cuda/` |
| **Salida métricas `_cpu`** | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase4_augment_roi_cpu/` (opcional) |
| **Análisis en notebook** | `…_augmentacion_roi_cuda.ipynb` (celdas finales actualizadas) |
| **Base de código** | `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` — `apply_roi_augment=True` en `multiclass_train_ds` (misma lógica que este experimento). |

### Ficha §7.1 — Fase 4, experimento `4_cuda` (augment ROI train, referencia Colab)

| Campo | Contenido |
|-------|-----------|
| **ID experimento** | `fase4_augment_roi_cuda_2026-05-14` |
| **Fase (número + nombre)** | Fase 4 — Augmentación geométrica suave en ROI multiclase (`_cuda`) |
| **Fecha cierre** | 2026-05-14 |
| **Responsable** | *(rellenar)* |
| **Hipótesis (1–2 frases)** | Rotación y escala pequeñas en el crop ROI multiclase **solo en train** mejoran `macro_dice_fg` / `macro_iou_fg` en test respecto al mejor candidato previo (Fase 3 run B), sin dañar el binario. |
| **Cambio vs Fase 3 run B** | Mismo flujo y hiperparámetros multiclase; `apply_roi_augment=True` en `multiclass_train_ds` (`apply_fase4_roi_geom_augment_uint8`). |
| **Ruta carpeta métricas** | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase4_augment_roi_cuda/` |
| **Binario — test** | Dice: **0,8710** \| IoU: **0,7714** \| loss: **0,2812** |
| **Multiclase — test** | `macro_dice_fg`: **0,2380** \| `macro_iou_fg`: **0,1415** \| loss: **2,2382** \| `pixel_accuracy`: **0,7502** |
| **Mejor `val_macro_dice_fg` (época)** | ≈ **0,2325** en época **22** |
| **Δ vs baseline (multiclase)** | `macro_dice_fg`: **+0,0486** \| `macro_iou_fg`: **+0,0289** |
| **Δ vs Fase 3 run B (multiclase)** | `macro_dice_fg`: **+0,0035** \| `macro_iou_fg`: **+0,0028** |
| **Dice por clase (test)** | T9 **0,1324**, T10 **0,0731**, T11 **0,1750**, L5 **0,1955** (vs run B: T10/T11 suben; T9/L5 bajan). |
| **ROI** | **198/198** `pred_binary`. |
| **Conclusión (2–4 frases)** | Los criterios §4.2 del plan (prioridad macro FG en test) se cumplen: mejora clara respecto al baseline y ganancia marginal respecto al run B con solo Fase 3. El binario se mantiene alineado con el baseline. Hay trade-off por vértebra (T9/L5 vs T10/T11) aceptable bajo la política de decisión acordada. |
| **Decisión** | **Adoptar** — mantener augment ROI en train como parte del procedimiento en el notebook vivo. **Fase 5** (multiseed) documentada aparte; sin cambio adicional al vivo tras Fase 5. |
| **Siguiente acción** | **Fase 6** (post-proceso ligero anatómico) u otras prioridades del plan; portar a Colab `03_…` cuando toque. |

### Métricas (test): baseline vs Fase 3 run B vs Fase 4 `_cuda`

| Métrica | Baseline | Fase 3 run B | Fase 4 `_cuda` |
|--------|----------|--------------|----------------|
| Binario — Dice | 0,8713 | 0,8710 | **0,8710** |
| Multiclase — `macro_dice_fg` | 0,1894 | 0,2345 | **0,2380** |
| Multiclase — `macro_iou_fg` | 0,1126 | 0,1387 | **0,1415** |

### Análisis (resumen)

- **Macro multiclase:** mejora respecto al **run B** y muy por encima del **baseline**; encaja con la hipótesis de Fase 4.
- **Binario:** sin regresión frente al baseline.
- **Por clase:** mejora fuerte en **T10** respecto a B; vigilar **T9** en futuros ajustes finos de augment si se prioriza esa región.

### Decisión (cierre Fase 4)

**Adoptar** la augmentación geométrica suave en ROI **train** multiclase. Quedó **integrada** en `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` (`apply_roi_augment=True` en entrenamiento multiclase).

---

## Fase 5 — Multiseed / robustez estadística (`mejorafase5_multiseed`)

**Estado:** tres variantes **`_cuda`** ejecutadas y cerradas (2026-05-15). **Decisión:** **«Prometedor; no consolidado»** — no se modifica el notebook vivo (sigue Fase 3 + Fase 4); la mejora macro frente al baseline **no** se replica de forma estable entre semillas.

| Campo | Valor |
|-------|--------|
| **Directorio de la fase** | `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase5_multiseed/` |
| **Notebooks** | `…_multiseed_cuda_seed42.ipynb`, `…_seed1337.ipynb`, `…_seed4242.ipynb` |
| **Regenerar** | `python notebooks/v3_local/mejoras/scripts/build_fase5_multiseed_notebooks.py` |
| **Celdas de análisis** | Tras regenerar, volver a ejecutar `python notebooks/v3_local/mejoras/scripts/patch_fase5_multiseed_analysis_cells.py` (inserta el bloque previo al **Registro de ejecución**). |
| **Referencias de comparación** | Baseline (`training_runs_cascade_v3/`), Fase 4 `_cuda` (`…_fase4_augment_roi_cuda/`). |

### Ficha §7.1 — Fase 5 (tres semillas; mismo split y pipeline vivo)

| Campo | Contenido |
|-------|------------|
| **IDs experimento** | `fase5_multiseed_cuda_s42_2026-05-15`, `fase5_multiseed_cuda_s1337_2026-05-15`, `fase5_multiseed_cuda_s4242_2026-05-15` |
| **Fase (número + nombre)** | Fase 5 — Robustez estadística / multiseed (`_cuda`) |
| **Fecha cierre** | 2026-05-15 |
| **Responsable** | *(rellenar)* |
| **Hipótesis (1–2 frases)** | Con el mismo split y el cascada vivo (Fase 3 + Fase 4), distintas `SEED` deberían mantener `macro_dice_fg` / `macro_iou_fg` en test en la **misma dirección** respecto al baseline y magnitud similar a Fase 4 `_cuda`. |
| **Cambio vs Fase 4** | Ninguno en algoritmo; solo `SEED`, `OUTPUT_DIR` y sufijos de checkpoints (semillas 42, 1337, 4242). |
| **Rutas carpetas métricas** | `…/training_runs_cascade_v3_fase5_multiseed_cuda_s42/`, `…_s1337/`, `…_s4242/` |
| **Multiclase — test (macro FG / IoU FG)** | **42:** 0,2294 / 0,1365 — **1337:** **0,2457** / **0,1503** — **4242:** 0,1737 / 0,1035 |
| **Binario — test (Dice)** | **42:** 0,8710 — **1337:** **0,8943** — **4242:** 0,8582 |
| **Mejor `val_macro_dice_fg` (época)** | **42:** ≈0,2226 (ép. 22) — **1337:** ≈0,2609 (ép. 19) — **4242:** ≈0,1685 (ép. 24) |
| **Δ vs baseline (macro multiclase)** | +0,0400 / +0,0563 / **−0,0157** (`macro_dice_fg`) |
| **Δ vs Fase 4 `_cuda` (macro multiclase)** | −0,0086 / **+0,0077** / **−0,0643** |
| **Dice por clase (test), foco T9–T11 / L5** | Ver tabla §7.2; **4242** tiene **T10 = 0** en test. |
| **ROI** | **198/198** `pred_binary` en los tres runs. |
| **Conclusión (2–4 frases)** | La dispersión entre semillas es **alta**: una semilla supera el mejor run Fase 4, otra queda cerca por debajo, una tercera cae por debajo del baseline con fallo severo en T10. El binario se comporta de forma acorde (excelente en 1337, estable en 42, algo peor en 4242). Esto **no** invalida la adopción previa de Fase 3 + Fase 4, pero **sí** impide afirmar **robustez estadística** con solo tres semillas. |
| **Decisión** | **«Prometedor; no consolidado»** (plan §3 Fase 5): **sin cambios de código** en el vivo; documentar varianza. **Fase 6** (post-proceso) **cerrada** — ver sección siguiente; **no** se adoptó post islas en el vivo. |
| **Siguiente acción** | **Fase 7** — estimador `last_visible` / clipping (`notebooks/07_colab_train_last_visible_estimator_and_clip_thoracolumbar_explained.ipynb`, carpeta `…_mejorafase7_auxiliares_rango_lastvis/`). |

### Métricas (test): baseline vs Fase 4 vs Fase 5 (tres semillas)

| Métrica | Baseline | Fase 4 `_cuda` | F5 `s42` | F5 `s1337` | F5 `s4242` |
|--------|----------|----------------|----------|------------|------------|
| Binario — Dice | 0,8713 | 0,8710 | 0,8710 | **0,8943** | 0,8582 |
| Multiclase — `macro_dice_fg` | 0,1894 | **0,2380** | 0,2294 | **0,2457** | 0,1737 |
| Multiclase — `macro_iou_fg` | 0,1126 | **0,1415** | 0,1365 | **0,1503** | 0,1035 |

### Análisis (resumen)

- **Replicación:** no demostrada; el rango de `macro_dice_fg` entre semillas es **amplio** (~0,174–0,246).
- **Fase 4 vs F5-42:** la pequeña baja (−0,0086) puede ser variación + entorno; **F5-1337** confirma que hay margen por encima de Fase 4 con otra semilla.
- **Riesgo:** **4242** muestra colapso de **T10** en test pese a ROI completo; conviene monitorizar torácico medio en experimentos futuros.

### Decisión (cierre Fase 5)

**«Prometedor; no consolidado».** Mantener el **notebook vivo** tal como está (Fase 3 + Fase 4). La Fase 5 aporta **evidencia de varianza**, no un nuevo ajuste obligatorio.

---

## Fase 6 — Post-proceso ligero anatómico (`mejorafase6_postproceso_ligero`)

**Estado:** variante **`_cuda`** ejecutada y cerrada (2026-05-15). **`FASE6_VERTICAL_MEDIAN_K = 0`** (solo eliminación de islas por clase, umbral **64** píxeles). **Decisión:** **No adoptar** integrar el post en el pipeline vivo.

| Campo | Valor |
|-------|--------|
| **Directorio de la fase** | `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase6_postproceso_ligero/` |
| **Notebooks** | `…_postproceso_ligero_cpu.ipynb`, `…_postproceso_ligero_cuda.ipynb` |
| **Regenerar** | `python notebooks/v3_local/mejoras/scripts/build_fase6_postproceso_notebooks.py` |
| **Celdas de análisis `_cuda`** | Tras regenerar: `python notebooks/v3_local/mejoras/scripts/patch_fase6_postproceso_analysis_cells.py` |
| **Salida métricas `_cuda`** | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase6_postproc_cuda/` |

### Ficha §7.1 — Fase 6, experimento `6_cuda` (islas mínimas, sin mediana vertical)

| Campo | Contenido |
|-------|-----------|
| **ID experimento** | `fase6_postproc_cuda_2026-05-15` |
| **Fase (número + nombre)** | Fase 6 — Post-proceso ligero (`_cuda`, `K=0`) |
| **Fecha cierre** | 2026-05-15 |
| **Responsable** | *(rellenar)* |
| **Hipótesis (1–2 frases)** | Eliminar componentes conexas pequeñas por etiqueta en la predicción multiclase (test) mejora `macro_dice_fg` / `macro_iou_fg` sin dañar el bloque torácico (T9–T12). |
| **Cambio vs vivo** | Mismo entrenamiento cascada (Fase 3 + Fase 4); solo segunda evaluación en test con `apply_fase6_postprocess=True` y CSV `*_fase6_post`. |
| **Ruta carpeta métricas** | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase6_postproc_cuda/` |
| **Binario — test** | Dice: **0,8710** \| IoU: **0,7714** (sin cambio por el post multiclase). |
| **Multiclase — test (sin post)** | `macro_dice_fg`: **0,2294** \| `macro_iou_fg`: **0,1360** \| `pixel_accuracy`: **0,7498** |
| **Multiclase — test (con post islas)** | `macro_dice_fg`: **0,2273** \| `macro_iou_fg`: **0,1349** \| `pixel_accuracy`: **0,7526** |
| **Δ post vs sin post (macro)** | `macro_dice_fg`: **−0,0021** \| `macro_iou_fg`: **−0,0011** |
| **Dice por clase (test), T7–T12** | T7 **−0,0049**, T8 **+0,0022**, T9 **−0,0099**, T10 **−0,0082**, T11 **−0,0031**, T12 **−0,0052** (sin post → con post). L5 estable (~0,1925). Detalle: `thoracolumbar_core_per_class_metrics.csv` vs `…_fase6_post.csv`. |
| **ROI** | **198/198** `pred_binary` (`thoracolumbar_core_binary_rois.csv`). |
| **Mediana vertical `K=3`** | **No evaluada** — con islas solas ya hubo baja de macro y de varias vértebras torácicas; no se priorizó segundo run. |
| **Conclusión (2–4 frases)** | El post de islas **no** cumple el criterio del plan (macro FG en test): empeora ligeramente el objetivo principal y el **torácico medio–bajo** en conjunto. No compensa activar el paso en inferencia. |
| **Decisión** | **No adoptar** — mantener inferencia multiclase **sin** este post-proceso en el notebook vivo. |
| **Siguiente acción** | **Fase 7** — `notebooks/07_colab_train_last_visible_estimator_and_clip_thoracolumbar_explained.ipynb` y carpeta `…_mejorafase7_auxiliares_rango_lastvis/` (README). |

### Métricas (test): multiclase sin post vs con post Fase 6 (`_cuda`)

| Métrica | Sin post | Con post (islas, `K=0`) |
|--------|----------|-------------------------|
| `macro_dice_fg` | 0,2294 | 0,2273 |
| `macro_iou_fg` | 0,1360 | 0,1349 |
| Binario — Dice | 0,8710 | 0,8710 |

### Análisis (resumen)

- **Macro multiclase:** el post **reduce** `macro_dice_fg` e `macro_iou_fg` en test frente a la misma corrida sin post.
- **Torácico:** T9–T12 tienden a **bajar** (T8 sube levemente); incumple la vigilancia §4.2 / informe 5.5 para un “solo ganancia lumbar”.
- **`K=3`:** no hace falta para cerrar la decisión con la evidencia actual.

### Decisión (cierre Fase 6)

**No adoptar** el post-proceso de islas (64 px, sin mediana) en el flujo de producción del proyecto. Los notebooks de la fase quedan como **referencia reproducible** del ensayo negativo.

---

## Fase 7 — Cascada + last_visible / clipping (`mejorafase7_auxiliares_rango_lastvis`)

**Estado:** variante **`_cuda`** ejecutada **completa** (2026-05-15). **Cascada (secc. 1–7)** y **last_visible + clipping (secc. 8)** cerradas. **Decisión:** **No adoptar**.

| Campo | Valor |
|-------|--------|
| **Directorio de la fase** | `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase7_auxiliares_rango_lastvis/` |
| **Notebooks** | `…_auxiliares_rango_lastvis_cpu.ipynb`, `…_auxiliares_rango_lastvis_cuda.ipynb` |
| **Regenerar** | `python notebooks/v3_local/mejoras/scripts/build_fase7_last_visible_notebooks.py` |
| **Celdas de análisis `_cuda`** | `python notebooks/v3_local/mejoras/scripts/patch_fase7_last_visible_analysis_cells.py` |
| **Salida cascada `_cuda`** | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase7_lastvis_cuda/` |
| **Salida last_visible `_cuda`** | `outputs/analysis_outputs_v3/training_runs_last_visible_fase7_cuda/` |

### Ficha §7.1 — Fase 7, experimento `7_cuda` (notebook autocontenido, run completo)

| Campo | Contenido |
|-------|-----------|
| **ID experimento** | `fase7_lastvis_cuda_2026-05-15` |
| **Fase (número + nombre)** | Fase 7 — Cascada + last_visible + clipping (`_cuda`) |
| **Fecha cierre** | 2026-05-15 |
| **Responsable** | *(rellenar)* |
| **Hipótesis (1–2 frases)** | En un solo notebook: entrenar cascada V3 vivo y, con esas máscaras/features, entrenar `last_visible_idx` y medir si el clipping por última vértebra mejora `macro_dice_fg` y reduce sobrepredicción (informe 5.6). |
| **Cambio vs vivo** | Mismo pipeline cascada que el baseline integrado (F3+F4); bloque §8 adaptado del notebook `07` con inferencia `UNetSmall` del baseline. |
| **Ruta carpeta métricas cascada** | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase7_lastvis_cuda/` |
| **Ruta carpeta last_visible** | `outputs/analysis_outputs_v3/training_runs_last_visible_fase7_cuda/` |
| **Binario — test** | Dice: **0,8710** \| IoU: **0,7714** \| loss: **0,2812** |
| **Multiclase — test (cascada)** | `macro_dice_fg`: **0,2258** \| `macro_iou_fg`: **0,1341** \| `pixel_accuracy`: **0,7470** |
| **Mejor `val_macro_dice_fg` (época)** | **0,2247** (época **22**) |
| **Δ vs Fase 4 `_cuda` (multiclase)** | `macro_dice_fg`: **−0,0122** \| `macro_iou_fg`: **−0,0074** (Fase 4: 0,2380 / 0,1415) |
| **Δ vs baseline histórico** | `macro_dice_fg`: **+0,0364** (baseline 0,1894) |
| **Dice por clase (test)** | T9 **0,0872**, T10 **0,0849**, T11 **0,1771**, L5 **0,1871** (T10 sube vs F4; T9/L5 bajan). |
| **Last visible — test** | Exact **0,25** \| Within-1 **0,425** \| MAE **2,025** \| Sobrepredicción **0,525** |
| **Clipping — `macro_dice_fg` test** | `raw` **0,2216** → `last_pred_clip` **0,2295** (+0,0079); `oracle_clip` **0,2491** (techo); **por debajo** de Fase 4 sin clip (0,2380) |
| **Conteo vértebras (test)** | Extra media: 2,48 → 1,35 con clip; missing: 0,65 → 1,38 (9/40 muestras empeoran) |
| **Conclusión (2–4 frases)** | El diseño autocontenido funciona de extremo a extremo. La cascada re-entrenada **no** supera Fase 4 en macro multiclase. El estimador last_visible tiene **baja** exactitud en test; el clipping `last_pred_clip` mejora marginalmente el macro del mismo run pero **no** alcanza el vivo ni compensa el aumento de vértebras faltantes. |
| **Decisión** | **No adoptar** — mantener pipeline **Fase 3 + Fase 4**; no integrar checkpoints fase7 ni auxiliar last_visible/clipping. |
| **Siguiente acción** | **Fase 8** (eficiencia) según plan §3. Reabrir last_visible solo con hipótesis nueva (más datos/features; objetivo within-1 test > 0,7). |

### Métricas (test): cascada Fase 7 vs referencias

| Métrica | Fase 7 `_cuda` | Fase 4 `_cuda` | Fase 6 `_cuda` (sin post) |
|--------|----------------|----------------|---------------------------|
| Binario — Dice | 0,8710 | 0,8710 | 0,8710 |
| `macro_dice_fg` | 0,2258 | **0,2380** | 0,2294 |
| `macro_iou_fg` | 0,1341 | **0,1415** | 0,1360 |

### Clipping (test, mismo run Fase 7)

| Modo | `macro_dice_fg` | `macro_iou_fg` |
|------|-----------------|----------------|
| `raw` | 0,2216 | 0,1315 |
| `last_pred_clip` | **0,2295** | 0,1367 |
| `oracle_clip` | 0,2491 | 0,1508 |
| `prev_range_clip` *(legacy)* | 0,2827 | 0,1737 |

### Análisis (resumen)

- **Cascada:** por debajo de Fase 4 en macro FG; binario estable; mejora T10 vs F4 pero T9 muy bajo (0,087).
- **Last visible:** generalización test insuficiente (exact 25 %).
- **Clipping:** mejora local +0,008 macro vs `raw` del run, pero **no** supera Fase 4 ni justifica despliegue con el estimador actual.

### Decisión (cierre Fase 7 — run 2026-05-15)

**No adoptar** cambios al procedimiento base. Los notebooks y CSV quedan como **referencia reproducible** del ensayo negativo (cascada autocontenida + auxiliar informe 5.6).

---

---

## Fase 8 — Eficiencia (`mejorafase8_eficiencia`)

**Estado:** variante **`_cuda`** ejecutada y cerrada (2026-05-15). **Decisión:** **No adoptar**.

| Campo | Valor |
|-------|--------|
| **ID experimento** | `fase8_eficiencia_cuda_2026-05-15` |
| **Directorio** | `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase8_eficiencia/` |
| **Regenerar** | `python notebooks/v3_local/mejoras/scripts/build_fase8_eficiencia_notebooks.py` |
| **Celdas análisis `_cuda`** | `python notebooks/v3_local/mejoras/scripts/patch_fase8_eficiencia_analysis_cells.py` |
| **Salida `_cuda`** | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase8_eficiencia_cuda/` |
| **Cambio vs vivo** | `IMG_SIZE` **384×192**; `UNET_BASE_CHANNELS=24`; mismas épocas; F3+F4 sin cambio lógico. |
| **Multiclase — test** | `macro_dice_fg`: **0,2157** \| `macro_iou_fg`: **0,1319** \| `pixel_accuracy`: **0,7495** |
| **Binario — test** | Dice: **0,8906** \| IoU: *(ver `binary_spine_test_metrics.csv`)* |
| **Δ vs Fase 4** | `macro_dice_fg`: **−0,0223** (umbral plan ≤ 0,01 → **no cumple**) |
| **Mejor val macro** | **0,2163** (ép. **20**) vs F4 **0,2325** (ep. 22) |
| **Dice por clase (test)** | T9 **0,1946**, T10 **0,1100**, T11 **0,1618**, L5 **0,2785** |
| **Conclusión** | Ahorro estructural de cómputo no compensa la pérdida de macro multiclase frente al vivo; binario algo mejor en este run no basta para adopción. |
| **Decisión** | **No adoptar** — mantener **512×256** y `base=32` en `train_spine_cascade_binary_to_thoracolumbar_v3.ipynb`. |

### Métricas (test): Fase 8 vs Fase 4

| Métrica | Fase 8 `_cuda` | Fase 4 `_cuda` |
|--------|----------------|----------------|
| `macro_dice_fg` | 0,2157 | **0,2380** |
| `macro_iou_fg` | 0,1319 | **0,1415** |
| Binario — Dice | 0,8906 | 0,8710 |

### Decisión (cierre Fase 8 — run 2026-05-15)

**No adoptar** el perfil de eficiencia (384×192 + U-Net `base=24`) en el pipeline de entrenamiento adoptado. El plan de mejoras cascada queda **cerrado** con procedimiento vivo **Fase 3 + Fase 4** a resolución completa.

---

## Resumen final — registro de experimentos

Cuadro generado desde `notebooks/v3_local/mejoras/experiment_registry.csv` (**editar el CSV primero**, luego ejecutar este script).

**Alcance:** solo experimentos con **fecha de ejecución** (`fecha` no vacía en el CSV). La fila baseline (`0`) no se duplica en la tabla de runs; sus métricas y el cuadro *Baseline vs notebook final vs mejor run* están arriba. **Columnas Δ** (tabla de runs): diferencia frente al baseline `0`. Variantes `_cpu` opcionales y pendientes siguen solo en el CSV. Por clase (T9–T11, L5): [tabla §7.2](#tabla-comparativa-72--inventario-acumulado-actualizar-al-cerrar-cada-experimento) (Fase 0).

**Pipeline adoptado al cierre del plan (fases 1–8):** notebook vivo `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` (Fase 3 + Fase 4). Las métricas test del notebook final se documentan con el run **`4_cuda`** (validación Colab). Checkpoints: `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase4_augment_roi_cuda/`.


### Baseline vs notebook final vs mejor run (test multiclase core)

| | ID | `macro_dice_fg` | `macro_iou_fg` | Δ vs baseline (dice / IoU) | Nota |
|---|-----|-----------------|----------------|----------------------------|------|
| **Baseline** | `0` | 0,1894 | 0,1126 | — | Cascada V3 core (`experiment_registry.csv`). |
| **Notebook final** (vivo F3+F4) | ref. `4_cuda` | 0,2380 | 0,1415 | +0,0486 / +0,0289 | `train_spine_cascade_binary_to_thoracolumbar_v3.ipynb`. |
| **Mejor run** (máx. `macro_dice_fg`) | `5_cuda_s1337` | 0,2457 | 0,1503 | +0,0563 / +0,0377 | Mayor macro entre runs ejecutados; decisión «prometedor no consolidado» (no sustituye al pipeline oficial). |


| ID | Fase | Fecha | Sem. | `macro_dice_fg` | `macro_iou_fg` | Δ dice | Δ IoU | Decisión | Hipótesis |
|----|------|-------|------|-----------------|----------------|--------|-------|----------|-----------|
| `1_cuda` | 1 | 2026-05-14 | 42 | 0,1807 | 0,1038 | -0,0087 | -0,0088 | **No adoptar** | Letterbox en crop multiclase vs resize base |
| `2_cuda` | 2 | 2026-05-14 | 42 | 0,1896 | 0,1104 | +0,0002 | -0,0022 | **No adoptar** | Pesos CE x1.40 en T7-T12 (class_id 7-12) tras estimate_multiclass_class_weights; Dice multiclase sin cambio |
| `3_cuda` | 3 | 2026-05-15 | 42 | 0,2323 | 0,1375 | +0,0429 | +0,0249 | Adoptar con condicion | LR multiclase x0.5 + CosineAnnealingLR (perfil Colab) |
| `3_cuda_repro_b` | 3 | 2026-05-14 | 42 | **0,2345** | **0,1387** | +0,0451 | +0,0261 | **Adoptar** | Repeticion run A + cudnn deterministic (verificacion) |
| `4_cuda` | 4 | 2026-05-14 | 42 | **0,2380** | **0,1415** | +0,0486 | +0,0289 | **Adoptar** | Augment ROI multiclase train (rot+/-4deg scale 0.98-1.02) sobre V3+Fase3 |
| `5_cuda_s1337` | 5 | 2026-05-15 | 1337 | 0,2457 | 0,1503 | +0,0563 | +0,0377 | Prometedor no consolidado | Multiseed misma pipeline F3+F4 semilla 1337 |
| `5_cuda_s42` | 5 | 2026-05-15 | 42 | 0,2294 | 0,1365 | +0,0400 | +0,0239 | Prometedor no consolidado | Multiseed misma pipeline F3+F4 semilla 42 (control vs Fase 4) |
| `5_cuda_s4242` | 5 | 2026-05-15 | 4242 | 0,1737 | 0,1035 | -0,0157 | -0,0091 | Prometedor no consolidado | Multiseed misma pipeline F3+F4 semilla 4242 |
| `6_cuda` | 6 | 2026-05-15 | 42 | 0,2294 | 0,1360 | +0,0400 | +0,0234 | **No adoptar** | Post-proceso islas por clase test (K=0 min 64px) sobre pipeline vivo F3+F4 |
| `7_cuda` | 7 | 2026-05-15 | 42 | 0,2258 | 0,1341 | +0,0364 | +0,0215 | **No adoptar** | Notebook autocontenido cascada+last_visible+clipping (bloque 07 adaptado) |
| `8_cuda` | 8 | 2026-05-15 | 42 | 0,2157 | 0,1319 | +0,0263 | +0,0193 | **No adoptar** | Resolucion 384x192 + UNet base=24 (eficiencia vs vivo 512x256 base=32) |

**Columnas solo en el CSV:** `directorio_mejoras`, `notebook_o_script`, `output_metricas`, `notas`. Hipótesis completa en el CSV si esta tabla es demasiado ancha en el preview del editor.

### Dónde está el detalle de cada fase

| Necesidad | Dónde mirar |
|-----------|-------------|
| Decisión rápida y métricas globales | Este documento (§7.2 + tabla anterior) o el CSV |
| Análisis narrativo, tablas test, gráficas del run | Notebook `*_cuda.ipynb` de la fase + celdas «Análisis e interpretación» |
| Entradas/salidas y regenerar notebooks | `README.md` en cada carpeta `…_mejorafaseN_…/` |
| Convenciones, criterios de adopción, glosario informe | `PLAN_ACCION_AJUSTES_MODELOS.md` |
| Exportar a Excel / filtrar por fase o decisión | `experiment_registry.csv` |


*Resumen final: 11 runs con fecha (de 18 filas en CSV). Generado por `sync_registry_to_resultados.py` (2026-05-15).*

---

*Tabla §7.2 en Fase 0 actualizada con la fila Fase 8. Resumen final alineado con `experiment_registry.csv`.*

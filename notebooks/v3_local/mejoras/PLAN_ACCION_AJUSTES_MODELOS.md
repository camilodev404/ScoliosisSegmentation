# Plan de acción: ajustes secuenciales a modelos entrenados

Documento de trabajo para implementar mejoras al procedimiento actual (pipeline cascada V3 y componentes del informe de avance), con extracción de métricas comparables y criterios de adopción.

Para **unificar redacción y extraer resultados hacia un informe** (PDF, Word, actas), usar el **glosario (§6)** y las **plantillas (§7)** junto con `notebooks/v3_local/mejoras/RESULTADOS_Y_DECISIONES_GENERAL.md`.

**Referencias alineadas:**

- Informe de avance entregado: secciones **5.5 Retos identificados** y **5.6 Próximos pasos**.
- Notebook en cascada: `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` (interpretación, análisis y sugerencias de mejora).
- Inspección ROI: `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3_inspection_ROI.ipynb`.

---

## 0. Organización y convenciones (definiciones operativas)

### 0.0 Raíz de este repositorio (`ScoliosisSegmentation`)

Las rutas de este documento son **relativas a la raíz del repo** (carpeta que contiene `README.md`, `data/`, `models/`, `notebooks/`, `outputs/`). El trabajo V3 local y el plan de mejoras están bajo `notebooks/v3_local/`.

| Rol | Ruta canónica |
|-----|----------------|
| Plan, registro CSV y carpetas por fase | `notebooks/v3_local/mejoras/` |
| Notebooks V3 (cascada, inspección, cobertura, etc.) | `notebooks/v3_local/` |
| Dataset (suele estar en `.gitignore`) | `data/Scoliosis_Dataset/` |
| Manifiestos y métricas experimentales V3 | `outputs/analysis_outputs_v3/` |
| Pesos entrenados | `models/` |

Los notebooks resuelven `ROOT` buscando el marcador `data/Scoliosis_Dataset`. Si hace falta forzar la raíz (p. ej. kernel en `/content` en Colab), usar **`MAIA_PROJECT_ROOT`** (nombre histórico en el código) con la ruta absoluta al clon, o `%cd` a esa raíz antes de la celda de imports. En **Colab** use el notebook **`_cuda`** de la fase para las decisiones comparables con el baseline. Los notebooks **`_cuda`** generados por `build_fase1_letterbox_notebooks.py` prueban además, antes que `MyDrive`, la ruta `/content/drive/Othercomputers/Mi portátil/ScoliosisSegmentation` (sincronización típica desde otro PC); la constante está en ese script por si el nombre de carpeta difiere.

### 0.1 Documentación por fase y referencias sin mover originales

- Por **cada fase de mejora** se crea un **directorio propio** bajo `notebooks/v3_local/mejoras/` con documentos independientes cuando aplique (README de la fase, notas de inspección, checklist de entradas/salidas, plantillas de comparación).
- Los **consumibles** (dataset bajo `data/Scoliosis_Dataset/`, manifiestos, notebooks base en `notebooks/v3_local/`, CSV históricos bajo `outputs/analysis_outputs_v3/`, pesos en `models/`, etc.) **no se mueven** de su ubicación original. En cada fase solo se **referencian** (documentado en el README de esa fase).
- Los **nuevos artefactos** que genere una mejora (nuevos runs, figuras, exportaciones) se documentan con su ruta. Para entrenamientos, usar una **subcarpeta nominal nueva** bajo `outputs/analysis_outputs_v3/` de forma que el baseline en `training_runs_cascade_v3/` **no se sobrescriba** salvo decisión explícita.

**Convención sugerida para carpetas de salida de métricas** (ejemplo):

`outputs/analysis_outputs_v3/training_runs_cascade_v3_fase<N>_<descripcion_corta>/`

Así se mantiene el histórico del baseline en `outputs/analysis_outputs_v3/training_runs_cascade_v3/` intacto y cada experimento queda **nominalmente alineado** con la fase en `notebooks/v3_local/mejoras/`.

### 0.2 Convención de nombres de directorios bajo `notebooks/v3_local/mejoras/`

- **Fase 0** (línea base asociada al notebook cascada V3 y diagnóstico):  
  `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase0_base/`
- **Fases N ≥ 1:**  
  `notebooks/v3_local/mejoras/<nombre_notebook_base>_mejorafase<N>_<nombre_descriptivo>/`  
  Ejemplos (ilustrativos):  
  - `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_letterbox_roi/`  
  - `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase2_pesos_t7_t12/`

El **nombre del notebook base** debe coincidir con el archivo principal que se modifica o del que se deriva el experimento (sin extensión `.ipynb` en el nombre del directorio, o con ella si el equipo prefiere uniformidad con el archivo; lo importante es **consistencia**).

### 0.3 Registro central de métricas y decisiones

- En **`notebooks/v3_local/mejoras/`** se mantiene el archivo **`RESULTADOS_Y_DECISIONES_GENERAL.md`**.
- Después de cada fase implementada o evaluada: añadir **métricas numéricas**, **análisis breve** y **decisión** (implementar al procedimiento / no / condicional). El análisis detallado puede vivir en el notebook o markdown del directorio de esa fase; el documento general **consolida** lo esencial para el historial del procedimiento de mejoras.
- El archivo `notebooks/v3_local/mejoras/experiment_registry.csv` sigue siendo el registro tabular complementario (filas por experimento).
- Para **redactar informes** (avances, entregas), usar el **glosario (§6)** con terminología unificada y la **plantilla por fase (§7)** para volcar resultados desde cada README o notebook al documento general y al PDF/Word.

### 0.4 Mapa de cambios y marcas dentro del notebook derivado

- Cada notebook de mejora debe incluir al inicio (y, donde aplique, **antes** de cada bloque de código modificado) un **mapa explícito**: qué partes del procedimiento difieren del notebook base, con qué finalidad y qué secciones de análisis pueden quedar obsoletas.
- Usar **marcas visibles** en markdown (p. ej. `### [FASE N — NOMBRE_CORTO]`) y comentarios en código (`# --- [FASE N / …]`) en todo punto donde la mejora **altere** flujo de datos, hiperparámetros o interpretación respecto al base.
- Objetivo: un revisor (o el propio autor tras semanas) debe localizar los cambios sin un diff manual largo contra el `.ipynb` original.

### 0.5 Variantes de ejecución `_cpu` y `_cuda` (obligatorio en fases con entrenamiento)

**Regla de equipo:** en **toda fase que incluya entrenamiento** (o re-entrenamiento) se entregan **siempre dos notebooks** (o un script de build que los genere ambos), con sufijos **`_cpu`** y **`_cuda`** en el nombre del archivo. No se considera cerrada la implementación de la fase si solo existe una variante.

| Sufijo | Entorno previsto | Rol |
|--------|------------------|-----|
| **`_cpu`** | Jupyter **local** en PC sin GPU viable o CUDA incompatible | Perfil liviano (menor resolución, `batch` pequeño, menos épocas). **Iteración**, depuración, demos y avances con poco hardware. |
| **`_cuda`** | **Google Colab**, servidor con GPU, o estación con CUDA compatible con el PyTorch del proyecto | Perfil “completo” o alineado al baseline (resolución y épocas comparables). **Referencias para decisión** frente a `training_runs_cascade_v3/` y para informes. |

Detalles operativos:

- Salidas y pesos **siempre separados** por sufijo (p. ej. carpetas `…_faseN_…_cpu/` vs `…_faseN_…_cuda/` bajo `outputs/analysis_outputs_v3/`, y `*_cpu_best.pt` vs `*_cuda_best.pt` en `models/`) para no pisar runs.
- Las métricas `_cpu` pueden **no** ser comparables 1:1 al baseline; deben documentarse como **exploratorias** salvo que se demuestre paridad de hiperparámetros.
- En **Colab** se ejecuta preferentemente el notebook **`_cuda`** tras montar Drive y fijar `MAIA_PROJECT_ROOT` o `%cd` a la raíz del clon (ver §0.0).
- El **script de build** que copia o parchea el base debe ser la **única fuente** de verdad para no divergir manualmente las dos copias.

### 0.6 Actualización de narrativa tras cada run

- Los textos de **interpretación**, **conclusiones** y bloques tipo “cómo interpretar…” copiados del notebook base deben **revisarse y reescribirse** cuando las métricas o curvas difieran; no debe quedar párrafo que contradiga los CSV del run actual.
- Flujo acordado: tras ejecutar, quien corre el notebook **actualiza** esas celdas (o comunica los números para actualización) y sincroniza `notebooks/v3_local/mejoras/RESULTADOS_Y_DECISIONES_GENERAL.md` + `notebooks/v3_local/mejoras/experiment_registry.csv`.
- No hay revisión automática periódica del notebook salvo **petición explícita**; el punto de control es humano tras cada entrenamiento o checkpoint de avance.

### 0.7 Autocontención de notebooks `mejorafase` y cadena desde el baseline

**Regla de equipo:** cada notebook bajo `notebooks/v3_local/mejoras/…_mejorafaseN_…/` debe **defenderse solo**: un único run (o variante `_cpu` / `_cuda`) no puede exigir haber ejecutado antes otros notebooks de **otras** carpetas `mejorafase*`.

| Paso | Acción |
|------|--------|
| 1. Experimentar | Ejecutar el par `_cpu` / `_cuda` de la fase N; métricas en carpeta nominal propia (`…_faseN_…_{cpu\|cuda}/`). |
| 2. Decidir | Adoptar / no / condicional → `RESULTADOS_Y_DECISIONES_GENERAL.md` + `experiment_registry.csv`. |
| 3. Integrar | Si se adopta, volcar el cambio en `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` (baseline vivo). |
| 4. Regenerar | Las fases **posteriores** pendientes se **vuelven a generar** con su `build_fase*_*.py` leyendo **ese baseline**, no el `.ipynb` experimental de una fase anterior. |

Detalles:

- El **script de build** de cada fase (p. ej. `build_fase4_augment_roi_notebooks.py`, `build_fase7_last_visible_notebooks.py`) es la **única fuente** del par `_cpu` / `_cuda`; no editar manualmente una copia y dejar la otra desfasada (§0.5).
- Los consumibles compartidos (dataset, manifiesto V3, splits históricos en `outputs/`) se **referencian** pero no sustituyen entrenar la cascada dentro del notebook cuando la fase lo incluye.
- **Fase 7** es cascada completa (secciones 1–7 del baseline) **más** bloque last_visible/clipping (sección 8); el notebook histórico `notebooks/07_colab_train_last_visible_estimator_and_clip_thoracolumbar_explained.ipynb` solo aporta el **bloque adaptado**, no es el documento de entrenamiento operativo.

---

## 1. Línea base reproducible

Antes de cualquier cambio, **congelar** referencia con mismos splits y mismos `unique_sample_id` en test.

| Componente | Ubicación artefactos V3 |
|------------|-------------------------|
| Cascada (binario + multiclase core) | `outputs/analysis_outputs_v3/training_runs_cascade_v3/` |
| Plano (control opcional) | `outputs/analysis_outputs_v3/training_runs_v3/` |

### Métricas mínimas por experimento

Coherente con el informe (5.5): no basar decisiones solo en `pixel_accuracy` cuando el modelo opera sobre ROI.

- Binario: `binary_spine_test_metrics.csv` (Dice, IoU, loss).
- Multiclase: `thoracolumbar_core_test_metrics.csv` (`macro_dice_fg`, `macro_iou_fg`, loss; `pixel_accuracy` solo como contexto).
- **Obligatorio:** `thoracolumbar_core_per_class_metrics.csv` (prioridad **T9–T12**, **L5**, torácico alto si afecta postproceso).
- ROI: `thoracolumbar_core_binary_rois.csv` (`roi_source`, tamaños de bbox).
- Entrenamiento: mejor época / curvas en `thoracolumbar_core_history.csv` (`val_macro_dice_fg`).
- **Opcional pero recomendable:** si el manifiesto o el split permiten estratificar, reportar las mismas métricas por subgrupo **Normal / Scoliosis** (coherente con el informe y con la variabilidad anatómica).

### Criterios de adopción al procedimiento

- **Adoptar** si: sube `macro_dice_fg` o `macro_iou_fg` en test y no empeora de forma grave el bloque acordado (p. ej. T9–T11 sin caída sistemática por encima de un umbral fijado por el equipo).
- **Adoptar con condición** si: mejora macro pero empeora algunas clases: documentar trade-off explícito (informe 5.5: trade-off postproceso / torácico).
- **No adoptar** si: solo mejora `pixel_accuracy` o baja `loss` sin ganancia en **macro FG** o en clases débiles priorizadas.

### 1.1 Metadatos mínimos por experimento (reproducibilidad)

En el README de la fase y/o en `experiment_registry.csv`, registrar al menos:

- **Commit de Git** (hash corto) del código con el que se entrenó o evaluó.
- **Versión de Python**, **PyTorch** y **CUDA/CPU** (una línea cada una).
- **Ruta exacta** del CSV de split usado (`thoracolumbar_core_split_train_val_test.csv` o copia nominal si se congela explícitamente).
- **Semilla** (`SEED`) y, si aplica, **época** del checkpoint evaluado.

Objetivo: poder **repetir o auditar** un número sin reconstruir el entorno a ciegas.

### 1.2 Umbrales sugeridos (calibrar por el equipo)

Valores de partida; pueden sustituirse por acuerdo grupal documentado en `notebooks/v3_local/mejoras/RESULTADOS_Y_DECISIONES_GENERAL.md`:

| Concepto | Umbral orientativo |
|----------|-------------------|
| Mejora clara en `macro_dice_fg` (test) | ≥ **+0,01** absoluto respecto al baseline de la fase anterior aceptada |
| Empeoramiento “aceptable” en una clase focal (p. ej. T10) | ≤ **−0,02** en Dice respecto a baseline, siempre que el objetivo de la fase lo justifique por escrito |
| Regresión “grave” en bloque acordado (p. ej. T9–T11 promedio) | > **−0,03** en Dice medio del bloque sin compensación en macro FG |
| Regresión en etapa **binaria** al tocar solo multiclase | Cualquier caída de Dice binario en test: **investigar primero** antes de adoptar (posible efecto colateral de ROI o de datos) |

### Registro de experimentos

- Nombrar runs con subcarpeta o sufijo (ej. `training_runs_cascade_v3_exp01_letterbox`).
- Mantener `notebooks/v3_local/mejoras/experiment_registry.csv` con: fecha, hipótesis, fase, directorio de documentación de la fase, hiperparámetros, rutas a CSV de salida, decisión (adoptar / no / pendiente).
- Actualizar en paralelo `notebooks/v3_local/mejoras/RESULTADOS_Y_DECISIONES_GENERAL.md` con el resumen de métricas y decisión (ver §0.3).

---

## 2. Mapa informe ↔ notebook cascada

| Informe (5.5 / 5.6) | Refuerzo desde notebook cascada V3 |
|---------------------|-------------------------------------|
| Sobrepredicción / parciales (5.5) | Mejor segmentación por clase y ROI estable alimenta estimadores y clipping. |
| Métricas engañosas con ROI (5.5) | Decisiones ancladas en macro Dice/IoU FG + por clase. |
| Postproceso vs torácico (5.5) | Post-proceso ligero que no destruya detalle torácico. |
| Coherencia anatómica (5.6) | Restricciones suaves de orden/continuidad; etapa `partial` cuando `core` esté estable. |
| Auxiliares rango / última vértebra (5.6) | Tras estabilizar multiclase en ROI y métricas por clase. |
| Augmentation / etiquetas (5.6) | Augmentación geométrica suave; letterbox; pérdida/muestreo T7–T12. |
| Eficiencia (5.6) | Fase tardía: resolución, modelo ligero, pruning/cuantización. |

---

## 3. Fases secuenciales (implementar → medir → decidir)

Cada fase debe producir: (a) **directorio bajo `notebooks/v3_local/mejoras/`** con documentación de la fase (§0.2), (b) cambio acotado en código/config en el notebook o script correspondiente, (c) CSV de métricas en carpeta de salida **sin pisar** el baseline salvo acuerdo, (d) fila en `notebooks/v3_local/mejoras/experiment_registry.csv`, (e) sección en `notebooks/v3_local/mejoras/RESULTADOS_Y_DECISIONES_GENERAL.md`, (f) análisis detallado en el notebook/markdown de la fase.

### Fase 0 — Diagnóstico y línea base (`mejorafase0_base`)

- **Directorio:** `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase0_base/` (README con inventario de consumibles y salidas).
- **Notebook de referencia:** `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` (no duplicar aquí; solo documentar).
- **Inspección visual ROI:** `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3_inspection_ROI.ipynb`.
- **Salida cualitativa:** notas en `NOTAS_INSPECCION_ROI.md` (opcional) dentro del directorio de la fase 0; patrones (ROI corta, descentrado dorsal, etc.) que informen la Fase 1.

### Fase 1 — Letterbox / padding fijo en recorte multiclase

- **Directorio sugerido:** `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_letterbox_roi/` (README + notebooks).
- **Notebooks generados:** `train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_letterbox_roi_cpu.ipynb` y `…_cuda.ipynb` (regenerables con `python notebooks/v3_local/mejoras/scripts/build_fase1_letterbox_notebooks.py` desde la raíz del repo, si evoluciona el cascada V3 base; ver §0.4–0.6).
- **Qué:** ROI → redimensionado con **relación de aspecto preservada** + padding al tamaño fijo de entrada, idéntico en train/val/test.
- **Por qué:** menos escalas inconsistentes; mejor diferenciación entre vértebras adyacentes (5.6); coherente con análisis T9–T11 en cascada V3.
- **Métricas:** mismos CSV + tabla Dice **T8–T12** vs baseline.
- **Adoptar si:** `macro_dice_fg` mejora o se mantiene y T10–T11 no empeoran más allá del umbral acordado (ej. 0.02 en Dice por clase).

### Fase 2 — Pérdida / muestreo consciente de clase (T7–T12)

- **Directorio sugerido:** `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase2_pesos_clases_t7_t12/`
- **Qué:** reponderar CE/Dice o focal suave en foreground hacia T7–T12 en el crop; opcional oversampling de batches con bajo Dice en esas clases (según `per_class_metrics` baseline).
- **Métricas:** `per_class_metrics` + `val_macro_dice_fg`; vigilar **L5** para no reintroducir colapso.
- **Adoptar si:** sube Dice medio en {T9, T10, T11} sin caída fuerte en L4–L5; macro FG no baja.

### Fase 3 — Optimización de entrenamiento multiclase

- **Directorio sugerido:** `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase3_scheduler_lr/`
- **Qué:** LR más bajo, scheduler (OneCycle/cosine), o congelar encoder al inicio; early stopping guiado por `val_macro_dice_fg` (no solo por loss).
- **Por qué:** reduce sobreajuste a bordes de ROI (trade-offs 5.5).
- **Adoptar si:** mejor val estable y test ≥ baseline con menor varianza entre semillas (complementa Fase 5).

### Fase 4 — Augmentación geométrica suave

- **Directorio sugerido:** `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase4_augmentacion_roi/`
- **Qué:** rotación y escala pequeñas en ROI, acotadas para radiografías.
- **Adoptar si:** mejora en test (macro FG) o en subgrupo Escoliosis si se estratifica la evaluación.

### Fase 5 — Robustez estadística

- **Directorio sugerido:** `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase5_multiseed/`
- **Regenerar notebooks:** `python notebooks/v3_local/mejoras/scripts/build_fase5_multiseed_notebooks.py` (tres variantes CUDA: semillas 42, 1337, 4242).
- **Qué:** repetir el mejor candidato de Fases 1–4 con 2–3 semillas (mismo split) o segundo split por grupo si hay tiempo.
- **Adoptar si:** la mejora se replica en dirección y magnitud; si solo una semilla gana → “prometedor, no consolidado”.

### Fase 6 — Post-proceso ligero anatómico

- **Directorio sugerido:** `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase6_postproceso_ligero/` (o el notebook/script que implemente solo inferencia/postproceso, manteniendo el mismo patrón de nombre).
- **Regenerar notebooks:** `python notebooks/v3_local/mejoras/scripts/build_fase6_postproceso_notebooks.py` (pares `_cpu` / `_cuda`; segunda evaluación test con CSV `*_fase6_post`).
- **Qué:** orden vertical suave, eliminación de islas mínimas por etiqueta, sin sustituir aún el pipeline completo del informe.
- **Métricas:** macro FG antes/después; vigilar torácico (5.5).
- **Adoptar si:** ganancia global o lumbar sin violar umbral de empeoramiento torácico acordado.

### Fase 7 — Auxiliares y parciales (informe 5.6)

- **Directorio:** `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase7_auxiliares_rango_lastvis/` (README + notebooks).
- **Regenerar notebooks:** `python notebooks/v3_local/mejoras/scripts/build_fase7_last_visible_notebooks.py` desde la raíz del repo (pares `_cpu` / `_cuda`; ver §0.7).
- **Fuente de generación:** `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` (cascada V3 vivo) + bloque last_visible adaptado desde `notebooks/07_colab_train_last_visible_estimator_and_clip_thoracolumbar_explained.ipynb` (solo lógica auxiliar; inferencia alineada a `UNetSmall` del baseline).
- **Qué (autocontenido):** en **un mismo notebook**: (1) entrenar **binario + multiclase** con el procedimiento baseline integrado; (2) entrenar o afinar estimador **`last_visible_idx`** sobre la cascada de **ese run**; (3) comparar **clipping** (`raw`, `range_pred_clip`, `last_pred_clip`, `oracle_clip`).
- **Salidas (no pisar baseline):**
  - Cascada: `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase7_lastvis_{cpu|cuda}/` (mismos CSV que §1: binario, multiclase core, splits, ROI).
  - Last visible: `outputs/analysis_outputs_v3/training_runs_last_visible_fase7_{cpu|cuda}/` (`last_visible_summary.csv`, `last_visible_clipping_metric_comparison.csv`, etc.).
- **Pesos en `models/` (generados por este run):** `binary_spine_cascade_fase7_lastvis_{cpu|cuda}_best.pt`, `thoracolumbar_core_cascade_fase7_lastvis_{cpu|cuda}_best.pt`, `last_visible_estimator_fase7_lastvis_{cpu|cuda}_best.pt`.
- **Métricas:** cascada — §1 (`macro_dice_fg`, `per_class_metrics`, binario test); last visible — Exact / Within-1 / MAE / overprediction + macro FG **post-clipping** (CSV en `LAST_OUTPUT_DIR`).
- **Adoptar si:** baja sobrepredicción o mejora macro FG post-clipping **sin** regresión inaceptable en la cascada del mismo experimento (binario + multiclase §1.2 / §4.2).

### Fase 8 — Eficiencia (al final, 5.6)

- **Directorio sugerido:** `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase8_eficiencia/` (o nombre del notebook base si la optimización aplica solo a inferencia).
- **Qué:** menor resolución, arquitectura más compacta, pruning/cuantización.
- **Adoptar si:** pérdida de métricas dentro de presupuesto acordado (ej. ≤1 punto en macro FG).

---

## 4. Ritmo de trabajo por iteración

1. Crear (o actualizar) el **directorio de la fase** bajo `notebooks/v3_local/mejoras/` con README de entradas/salidas y rutas a consumibles **sin moverlos**.
2. Una **hipótesis** por fase (evitar mezclar letterbox + focal + scheduler en el primer experimento).
3. Misma semilla y mismo código salvo el parámetro bajo prueba.
4. Cierre: tabla **baseline | experimento | Δ** para métricas globales + subtabla **T9–T11 y L5**; copiar resumen a `notebooks/v3_local/mejoras/RESULTADOS_Y_DECISIONES_GENERAL.md` y fila en `notebooks/v3_local/mejoras/experiment_registry.csv`.

### 4.1 Checklist de cierre (definición de “hecho”)

Antes de dar por cerrada una fase, verificar:

- [ ] README de la fase actualizado (entradas, salidas, metadatos §1.1).
- [ ] Métricas globales + por clase críticas copiadas al documento general.
- [ ] Decisión explícita (adoptar / no / condicional) con una frase de motivo.
- [ ] Si la fase tocó código de entrenamiento: **diff o rama Git** identificable desde el README.
- [ ] **Plantilla §7** (o equivalente) rellenada en el README de la fase y resumen pegado en `notebooks/v3_local/mejoras/RESULTADOS_Y_DECISIONES_GENERAL.md`.

### 4.2 Conflictos entre métricas (prioridad)

Si varias métricas discrepan, aplicar este orden de prioridad para la **decisión de adopción** en segmentación toracolumbar **core**:

1. `macro_dice_fg` y `macro_iou_fg` en test (foreground).
2. Dice por clase en **T9–T12** y **L5** (y cualquier clase marcada como objetivo en el README de la fase).
3. Estabilidad de `val_macro_dice_fg` en el historial (sobreajuste).
4. `pixel_accuracy` multiclase y `loss` solo como contexto, no como decisión principal.
5. Si un cambio es **solo multiclase** pero altera la ROI o el binario: reevaluar **binario** en test; una regresión binaria exige análisis causal antes de adoptar.

### 4.3 Uso de la plantilla y del glosario al escribir informes

- Copiar la **tabla de la §7.1** al inicio de la sección de resultados de cada fase en `notebooks/v3_local/mejoras/RESULTADOS_Y_DECISIONES_GENERAL.md` (y ampliar en el README de la fase con figuras o tablas extensas).
- Para **comparativos** entre experimentos, usar la **tabla §7.2** (una fila por experimento o por variante).
- Revisar el **glosario §6** antes de publicar: evita mezclar nombres de métricas o siglas distintas entre capítulos.

---

## 5. Calidad de datos (paralelo, 5.5 / 5.6)

- Aislar o corregir imágenes con `needs_annotation_review` (sub-manifiesto).
- Repetir el mejor run ganador **con y sin** esas muestras para acotar el techo de mejora.

---

## 6. Glosario y términos (redacción de informes)

Uso recomendado: **misma denominación** en notebooks, `notebooks/v3_local/mejoras/RESULTADOS_Y_DECISIONES_GENERAL.md` y documentos formales.

| Término / sigla | Definición breve |
|-----------------|------------------|
| **ROI** | Región de interés; aquí, recorte de la radiografía centrado en la columna (procedente de máscara binaria predicha o GT según etapa). |
| **Binario / etapa binaria** | Segmentación columna vs fondo; produce máscara o bbox para recortar. |
| **Multiclase / etapa multiclase** | Segmentación con etiquetas T1–T12 y L1–L5 (más fondo); en este proyecto suele operar **dentro de la ROI**. |
| **Core** | Subconjunto de imágenes con cobertura toracolumbar “completa” según manifiesto (`usable_for_thoracolumbar_core`). |
| **Partial** | Subconjunto con anatomía parcialmente visible; mayor desafío de sobre-predicción. |
| **GT** | *Ground truth*; anotación de referencia. |
| **FG** | *Foreground*; en métricas `macro_*_fg`, promedio sobre **clases de vértebra** (excluye típicamente fondo). |
| **`macro_dice_fg`** | Dice macro promediado sobre clases de primer plano (vértebras); métrica principal para decidir mejoras multiclase en ROI. |
| **`macro_iou_fg`** | IoU macro análogo en foreground. |
| **`pixel_accuracy`** | Proporción de píxeles correctamente clasificados; **puede ser engañosa** con poco fondo en ROI (informe 5.5). |
| **`per_class_metrics`** | Archivo CSV con Dice/IoU por etiqueta; obligatorio para detectar regresiones localizadas (p. ej. T9–T11). |
| **`unique_sample_id`** | Identificador de muestra en tablas de split y ROI; clave para trazabilidad. |
| **Split por grupos** | Partición train/val/test sin mezclar pacientes/grupos entre particiones (reduce fuga de información). |
| **Cascada** | Pipeline en dos etapas: binario → multiclase sobre ROI (u otras etapas posteriores del informe). |
| **Plano** (baseline alternativo) | Entrenamiento multiclase sobre imagen completa sin ROI predicha (carpeta `training_runs_v3` en el plan V3). |
| **Clipping anatómico** | Recorte de etiquetas fuera del rango visible estimado (última vértebra / rango visible en el informe). |
| **Last visible / última vértebra visible** | Estimador auxiliar que delimita el límite inferior de anatomía presente. |
| **Postproceso v2** | Reglas anatómicas de corrección sobre la máscara multiclase (orden, discontinuidades, etc., según notebooks del proyecto). |
| **Letterbox** | Redimensionar manteniendo proporción y rellenar bordes hasta un tamaño fijo de tensor. |
| **CE** | *Cross-entropy*; pérdida de clasificación por píxel. |
| **Focal** | Pérdida focal; enfatiza ejemplos difíciles (clases raras o mal clasificadas). |
| **Umbral / threshold** | Valor de probabilidad para binarizar salidas (p. ej. máscara binaria). |
| **`pos_weight`** | Peso de la clase positiva en BCE/CE para compensar desbalance fondo/columna. |
| **Época** | Iteración completa sobre el conjunto de entrenamiento; citar época del checkpoint si no es la última. |

---

## 7. Plantillas para extraer resultados (→ informe / acta)

Copiar y pegar en el README de cada fase y, en forma resumida, en `notebooks/v3_local/mejoras/RESULTADOS_Y_DECISIONES_GENERAL.md`. Las cifras son **ejemplo**; sustituir tras cada run.

### 7.1 Ficha de una fase o experimento (hipótesis → resultados → conclusión)

| Campo | Contenido |
|-------|-----------|
| **ID experimento** | ej. `fase1_letterbox_2026-05-20_run01` |
| **Fase (número + nombre)** | ej. Fase 1 — Letterbox ROI |
| **Fecha cierre** | AAAA-MM-DD |
| **Responsable** | Nombre o iniciales |
| **Hipótesis (1–2 frases)** | Qué se espera mejorar y por qué. |
| **Cambio vs baseline** | Lista breve de knobs (código, hiperparámetros, datos). |
| **Ruta directorio documentación** | `notebooks/v3_local/mejoras/..._mejorafaseN_.../` |
| **Ruta carpeta métricas generadas** | `outputs/analysis_outputs_v3/.../` |
| **Git commit** | Hash corto |
| **Entorno** | Python x.y, PyTorch x.y, CUDA sí/no |
| **Binario — test** | Dice: … \| IoU: … \| loss: … |
| **Multiclase — test** | `macro_dice_fg`: … \| `macro_iou_fg`: … \| loss: … \| `pixel_accuracy`: … (contexto) |
| **Mejor `val_macro_dice_fg` (época)** | Valor y n.º de época |
| **Δ vs baseline (multiclase)** | Δ `macro_dice_fg`: … \| Δ `macro_iou_fg`: … |
| **Dice por clase (test)** | Tabla o “ver CSV”: T9 …, T10 …, T11 …, L5 … (y otras si aplica) |
| **ROI / calidad recorte** | Resumen: `% pred_binary`, bbox medios, notas cualitativas de inspección |
| **Estratificación (opcional)** | Normal: … \| Scoliosis: … (mismas métricas si se calculan) |
| **Riesgos / efectos secundarios** | ej. regresión torácica, inestabilidad val, tiempo de entrenamiento |
| **Conclusión (2–4 frases)** | Lectura integrada de lo anterior. |
| **Decisión** | **Adoptar** / **No adoptar** / **Adoptar con condición** — condición explícita |
| **Siguiente acción** | ej. fusionar a rama principal, iterar Fase 2, descartar |

### 7.2 Tabla comparativa multi-experimento (para sección “Resultados” del informe)

Una fila por run o variante; las columnas pueden exportarse a Excel desde el mismo contenido.

| ID | Fase | `macro_dice_fg` (test) | `macro_iou_fg` (test) | Dice T9 | Dice T10 | Dice T11 | Dice L5 | Binario Dice (test) | Nota breve |
|----|------|------------------------|------------------------|---------|----------|----------|---------|----------------------|------------|
| baseline | 0 | … | … | … | … | … | … | … | Referencia cascada V3 |
| … | 1 | … | … | … | … | … | … | … | … |

### 7.3 Párrafos tipo para “Discusión” o “Análisis” (rellenar con números)

Texto guía (sustituir corchetes):

1. *Resumen cuantitativo:* «Respecto al baseline (macro Dice FG [valor_baseline]), el experimento [ID] alcanzó [valor_exp] (Δ = [delta]), con macro IoU FG de […]. La etapa binaria [se mantuvo / mejoró / empeoró] con Dice […].»

2. *Por clase:* «Las mayores variaciones se observaron en [clases], con énfasis en el bloque torácico medio (T9–T11) donde [descripción]. La región lumbar (p. ej. L5) […].»

3. *Limitaciones:* «Las métricas globales de precisión por píxel no se usaron como criterio principal debido al menor fondo en ROI ([cita informe 5.5 / plan]).»

4. *Decisión:* «Por [criterios §1.2 y §4.2], se [adopta / no adopta / adopta con condición X] la modificación al procedimiento.»

---

## 8. Control de versiones de este documento

| Versión | Cambio |
|---------|--------|
| 1.4 | §0.7 autocontención `mejorafase` y cadena baseline → regenerar; §3 Fase 7 como notebook autocontenido (cascada + last_visible), salidas duales, `build_fase7_last_visible_notebooks.py`. |
| 1.3 | Alineación a repo `ScoliosisSegmentation`: §0.0 (rutas `notebooks/v3_local/`, `data/`, `outputs/analysis_outputs_v3/`), obligatoriedad dual `_cpu`/`_cuda` en §0.5, rutas de documentación y scripts actualizadas. |
| 1.2 | Glosario para informes (§6), plantillas ficha / tabla comparativa / párrafos tipo (§7), vínculo desde §0.3 y checklist §4.1; reordenación de control de versiones a §8. |
| 1.1 | Metadatos de reproducibilidad (§1.1), umbrales sugeridos (§1.2), convención de carpetas de salida bajo `outputs/analysis_outputs_v3/` (§0.1), métricas estratificadas opcionales, directorios sugeridos fases 2–8, checklist DoD (§4.1), prioridad ante conflictos de métricas (§4.2). |
| 1.0 | Convenciones de carpeta `mejoras/` (histórico MAIA-PROYECTO), fase 0, registro central y fases 0–8. |

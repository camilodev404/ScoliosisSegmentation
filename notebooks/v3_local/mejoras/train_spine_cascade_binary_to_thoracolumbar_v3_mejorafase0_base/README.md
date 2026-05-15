# Fase 0 — Línea base y diagnóstico (`mejorafase0_base`)

Este directorio agrupa la **documentación y notas** de la fase 0 del plan de mejoras, y una **réplica congelada** del notebook cascada V3 para fines académicos (comparar siempre contra el mismo artefacto `.ipynb`). Los artefactos de entrenamiento (CSV, `.pt`) siguen en `outputs/` según cada run.

**Siguiente:** **Fase 7** (estimador última vértebra visible / clipping) — `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase7_auxiliares_rango_lastvis/README.md` y `notebooks/07_colab_train_last_visible_estimator_and_clip_thoracolumbar_explained.ipynb`. **Fases 5–6 cerradas** — ver `RESULTADOS_Y_DECISIONES_GENERAL.md`.

## Notebook baseline en esta carpeta (réplica, no sustituye al de trabajo)

| Rol | Ruta |
|-----|------|
| **Baseline congelado (copia)** | `train_spine_cascade_binary_to_thoracolumbar_v3_baseline.ipynb` — copia del cascada V3 local para tesis o informes; **no** es el que parchean los scripts `build_fase*.py`. Actualízala solo cuando quieras **renovar el ancla** (p. ej. tras etiquetar un release en git). **Nota:** si el notebook **vivo** acumula mejoras (p. ej. Fase 3 ya integrada), esta copia puede quedar **por detrás** del vivo hasta que vuelvas a exportarla a propósito. |
| **Notebook de trabajo (única fuente para builds)** | `../train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` — aquí conviene **fusionar** mejoras adoptadas (p. ej. Fase 3) vía commit/PR para que las fases posteriores sigan generándose desde un solo archivo. |

### Flujo recomendado: ¿PR al base o “nueva versión” como base de experimentos?

- **Congelar baseline académico** en esta carpeta (`*_baseline.ipynb`): siempre puedes citar en memoria o artículo *exactamente* el código que produjo `training_runs_cascade_v3/` frente a carpetas `…_faseN_…/`, sin ambigüedad.
- **Integrar lo adoptado en el notebook de trabajo** (`v3_local/train_spine_cascade…_v3.ipynb`): una sola ruta para `build_fase*.py`, menos bifurcaciones y menos riesgo de que un experimento “nuevo base” se desincronice del Colab `03_…` oficial.
- **No sustituir** el baseline `.ipynb` por cada mejora aprobada: si cada fase apuntara a un “nuevo base” distinto, el término “baseline” en tablas de métricas seguiría siendo `training_runs_cascade_v3`, pero el código ya no sería comparable línea a línea con el snapshot de tesis. Mejor: **métricas baseline fijas en disco** + **snapshot `.ipynb` fijo en fase0** + **código vivo** que evoluciona con PRs.

## Convención de rutas

Todas las rutas siguientes son **relativas a la raíz del repo** (carpeta con `README.md`, `data/`, `models/`, `outputs/`, `notebooks/`).

## Base en la rama oficial (notebooks numerados `01`–`09`)

En el repo, la secuencia **01 → 09** describe el pipeline completo. Para **`mejorafase0_base`** y las fases de mejora que parten de la **cascada** (binario → ROI → multiclase), la referencia correcta en esa línea oficial es:

| Notebook | Rol respecto a las mejoras en cascada |
|----------|----------------------------------------|
| `01_colab_thoracolumbar_coverage_strategy_clean.ipynb` | **Precede** al entrenamiento: cobertura, manifiesto y revisión de datos; no es la base del entrenamiento en cascada. |
| `02_colab_train_spine_binary_and_thoracolumbar.ipynb` | Entrenamiento **plano** (binario + multiclase en imagen completa); **alternativa** al flujo en cascada, no el notebook base de `mejorafase0_base`. |
| **`03_colab_train_spine_cascade_binary_to_thoracolumbar_explained.ipynb`** | **Base oficial en la rama** para el mismo procedimiento que documenta esta fase: cascada explicada (binario → ROI → multiclase). Es el que debe tomarse como ancla conceptual frente a `main`. |
| `04`–`08` | Inferencia, postproceso, estimadores y pipeline final: **downstream** del entrenamiento en cascada; sirven para cerrar el proyecto, no como punto de partida de experimentos de entrenamiento en cascada. |
| `09_colab_final_project_summary_thoracolumbar_explained.ipynb` | Resumen final; tampoco sustituye al 03 como base de experimentación de modelos en cascada. |

**Conclusión:** para alinear documentación y criterios con **lo que hay en la rama**, el notebook a citar como **base del procedimiento en cascada** es **`notebooks/03_colab_train_spine_cascade_binary_to_thoracolumbar_explained.ipynb`**.

## Notebook donde ejecutáis las mejoras (línea V3 local)

La carpeta `mejoras/` y el registro de métricas están acopladas a la copia de trabajo bajo `notebooks/v3_local/`. Los scripts de build leen **`train_spine_cascade_binary_to_thoracolumbar_v3.ipynb`**; la copia en `mejorafase0_base/` sirve de **ancla documental** sin romper rutas.

| Rol | Ruta |
|-----|------|
| Entrenamiento cascada V3 (binario + multiclase core), editable | `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` |
| Misma lógica, copia para baseline académico | `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase0_base/train_spine_cascade_binary_to_thoracolumbar_v3_baseline.ipynb` |
| Inspección visual de ROI (diagnóstico) | `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3_inspection_ROI.ipynb` |

## Entradas consumibles (no mover)

| Recurso | Ruta |
|---------|------|
| Manifiesto thoracolumbar V3 | `outputs/analysis_outputs_v3/training_manifest_thoracolumbar_v3.csv` |
| Índice dataset | `data/Scoliosis_Dataset/indice_dataset.csv` |
| Diccionario de etiquetas T1–T12, L1–L5 | `data/Scoliosis_Dataset/diccionario_etiquetas_T1_T12_L1_L5.json` |

## Salidas generadas por el notebook de referencia (ubicación original)

Los entrenamientos ya ejecutados escriben por defecto en:

`outputs/analysis_outputs_v3/training_runs_cascade_v3/`

Archivos típicos a citar en análisis y en `notebooks/v3_local/mejoras/RESULTADOS_Y_DECISIONES_GENERAL.md`:

| Artefacto | Ruta |
|-----------|------|
| Métricas test binario | `outputs/analysis_outputs_v3/training_runs_cascade_v3/binary_spine_test_metrics.csv` |
| Métricas test multiclase core | `outputs/analysis_outputs_v3/training_runs_cascade_v3/thoracolumbar_core_test_metrics.csv` |
| Métricas por clase | `outputs/analysis_outputs_v3/training_runs_cascade_v3/thoracolumbar_core_per_class_metrics.csv` |
| ROI binarias por muestra | `outputs/analysis_outputs_v3/training_runs_cascade_v3/thoracolumbar_core_binary_rois.csv` |
| Historial multiclase | `outputs/analysis_outputs_v3/training_runs_cascade_v3/thoracolumbar_core_history.csv` |
| Particiones train/val/test | `outputs/analysis_outputs_v3/training_runs_cascade_v3/thoracolumbar_core_split_train_val_test.csv` |
| Pesos modelo binario (referencia) | `models/binary_spine_cascade_best.pt` |
| Pesos multiclase cascada core | `models/thoracolumbar_core_cascade_best.pt` |

## Documentos propios de esta fase (opcional)

Puedes añadir aquí (sin mover datos del dataset):

- `NOTAS_INSPECCION_ROI.md` — hallazgos cualitativos al ejecutar el notebook de inspección.
- Capturas exportadas — si se guardan, usar subcarpeta `figuras/` **dentro de este directorio** para no mezclar con salidas de entrenamiento.

## Próximo paso del plan

Tras cerrar el diagnóstico ROI, la **Fase 1** tiene su directorio bajo `notebooks/v3_local/mejoras/` con nombre del tipo:

`train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_<nombre_descriptivo>/`

Para **Fase 5** (multiseed, **cerrada**), ver `../train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase5_multiseed/README.md`. **Fase 6** (post-proceso): `../train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase6_postproceso_ligero/README.md` y `../PLAN_ACCION_AJUSTES_MODELOS.md` §3.

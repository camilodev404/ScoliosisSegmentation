# Fase 0 — Línea base y diagnóstico (`mejorafase0_base`)

Este directorio agrupa la **documentación y notas** de la fase 0 del plan de mejoras. No sustituye ni mueve el notebook de entrenamiento ni los artefactos generados por el pipeline; solo **referencia** rutas del repositorio.

**Siguiente:** Fase 1 (letterbox) — `mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_letterbox_roi/README.md`.

## Convención de rutas

Todas las rutas siguientes son **relativas a la raíz del proyecto** (`MAIA-PROYECTO/`), salvo que se indique lo contrario.

## Notebook y procedimiento de referencia (no mover)

| Rol | Ruta |
|-----|------|
| Entrenamiento cascada V3 (binario + multiclase core) | `train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` |
| Inspección visual de ROI (diagnóstico) | `train_spine_cascade_binary_to_thoracolumbar_v3_inspection_ROI.ipynb` |

## Entradas consumibles (no mover)

| Recurso | Ruta |
|---------|------|
| Manifiesto thoracolumbar V3 | `analysis_outputs_v3/training_manifest_thoracolumbar_v3.csv` |
| Índice dataset V3 | `Scoliosis_Dataset_V3/Scoliosis_Dataset/indice_dataset.csv` |
| Diccionario de etiquetas T1–T12, L1–L5 | `Scoliosis_Dataset_V3/Scoliosis_Dataset/diccionario_etiquetas_T1_T12_L1_L5.json` |

## Salidas generadas por el notebook de referencia (ubicación original)

Los entrenamientos ya ejecutados escriben por defecto en:

`analysis_outputs_v3/training_runs_cascade_v3/`

Archivos típicos a citar en análisis y en `RESULTADOS_Y_DECISIONES_GENERAL.md`:

| Artefacto | Ruta |
|-----------|------|
| Métricas test binario | `analysis_outputs_v3/training_runs_cascade_v3/binary_spine_test_metrics.csv` |
| Métricas test multiclase core | `analysis_outputs_v3/training_runs_cascade_v3/thoracolumbar_core_test_metrics.csv` |
| Métricas por clase | `analysis_outputs_v3/training_runs_cascade_v3/thoracolumbar_core_per_class_metrics.csv` |
| ROI binarias por muestra | `analysis_outputs_v3/training_runs_cascade_v3/thoracolumbar_core_binary_rois.csv` |
| Historial multiclase | `analysis_outputs_v3/training_runs_cascade_v3/thoracolumbar_core_history.csv` |
| Particiones train/val/test | `analysis_outputs_v3/training_runs_cascade_v3/thoracolumbar_core_split_train_val_test.csv` |
| Pesos modelo binario (referencia) | `models/binary_spine_cascade_best.pt` |
| Pesos multiclase cascada core | `models/thoracolumbar_core_cascade_best.pt` |

## Documentos propios de esta fase (opcional)

Puedes añadir aquí (sin mover datos del dataset):

- `NOTAS_INSPECCION_ROI.md` — hallazgos cualitativos al ejecutar el notebook de inspección.
- Capturas exportadas — si se guardan, usar subcarpeta `figuras/` **dentro de este directorio** para no mezclar con salidas de entrenamiento.

## Próximo paso del plan

Tras cerrar el diagnóstico ROI, la **Fase 1** tendrá su propio directorio bajo `mejoras/`, con nombre del tipo:

`train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_<nombre_descriptivo>/`

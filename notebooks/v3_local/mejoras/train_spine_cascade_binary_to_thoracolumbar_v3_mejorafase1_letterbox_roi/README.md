# Fase 1 — Letterbox en ROI multiclase (`mejorafase1_letterbox_roi`)

## Objetivo

Reducir la **deformación por estiramiento** al pasar el recorte de columna a la resolución fija del modelo multiclase, usando **letterbox** (escala uniforme + padding centrado). La máscara se rellena con `IGNORE_INDEX` (255) para que esos píxeles no entren en la `CrossEntropyLoss`.

## Esquema dual `_cpu` / `_cuda` (obligatorio)

| Archivo | Entorno | Rol |
|---------|---------|-----|
| `train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_letterbox_roi_cpu.ipynb` | Jupyter **local** (sin GPU o CUDA incompatible) | Perfil liviano: menos tiempo; métricas con salvedad frente al baseline. |
| `train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_letterbox_roi_cuda.ipynb` | **Google Colab** u otra máquina con GPU y PyTorch compatible | Perfil comparable al cascada V3 base (resolución y épocas); **referencia para decisión** vs `training_runs_cascade_v3/`. |

Cada notebook incluye al inicio un **mapa de cambios** respecto al V3 base y marcas **`[FASE 1]`** en markdown y código donde se implementa el letterbox. Tras cada entrenamiento, **actualiza** las celdas de interpretación y conclusiones para que coincidan con los CSV generados.

### Raíz del repo (local y Colab)

La primera celda de código resuelve la raíz con `_resolve_maia_project_root` (marcador `data/Scoliosis_Dataset`). En **Colab**, el kernel suele arrancar en `/content`: puedes definir antes  
`os.environ['MAIA_PROJECT_ROOT'] = r'/ruta/absoluta/al/clon'`  
o ejecutar `%cd` a la raíz del repo, luego abrir el notebook **`_cuda`**.

**`_cuda` y sincronización “Other computers”:** el notebook `*_cuda.ipynb` prueba por defecto (antes que `MyDrive/Colab Notebooks`) la ruta:

`/content/drive/Othercomputers/Mi portátil/ScoliosisSegmentation`

(es decir, base `.../Othercomputers/Mi portátil` y carpeta del proyecto `ScoliosisSegmentation`). Si el nombre de la carpeta del portátil en Drive no coincide exactamente, usa `MAIA_PROJECT_ROOT` o edita la constante `COLAB_CUDA_PROJECT_ROOT` en `notebooks/v3_local/mejoras/scripts/build_fase1_letterbox_notebooks.py` y regenera.

**Regenerar ambos notebooks** desde la **raíz del repo**:

```bash
python notebooks/v3_local/mejoras/scripts/build_fase1_letterbox_notebooks.py
```

(El script `build_fase1_letterbox_notebook.py` sin **s** delega al mismo flujo.)

## Salidas (no pisan el baseline)

| Variante | Métricas / historiales | Pesos sugeridos |
|----------|------------------------|-----------------|
| `_cpu` | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase1_letterbox_cpu/` | `models/binary_spine_cascade_fase1_letterbox_cpu_best.pt`, `models/thoracolumbar_core_cascade_fase1_letterbox_cpu_best.pt` |
| `_cuda` | `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase1_letterbox_cuda/` | `models/binary_spine_cascade_fase1_letterbox_cuda_best.pt`, `models/thoracolumbar_core_cascade_fase1_letterbox_cuda_best.pt` |

La variante `_cpu` prioriza **tiempo**; las métricas no son comparables 1:1 al baseline 512×256 sin salvedad. La variante `_cuda` es la referencia para comparar con `outputs/analysis_outputs_v3/training_runs_cascade_v3/`.

## Entradas (no mover; solo lectura)

Igual que el notebook cascada V3: manifiesto, índice bajo `data/Scoliosis_Dataset/`, JSON de etiquetas (ver README fase 0).

## Tras ejecutar

1. Rellenar **§7.1** y filas en **§7.2** en `notebooks/v3_local/mejoras/RESULTADOS_Y_DECISIONES_GENERAL.md` (una entrada por variante si corres ambas).
2. Filas en `notebooks/v3_local/mejoras/experiment_registry.csv` (`1_cpu` / `1_cuda` o equivalente).
3. Comparar la variante `_cuda` con baseline en `outputs/analysis_outputs_v3/training_runs_cascade_v3/` (plan §1.2 y §4.2); documentar la `_cpu` como apoyo si aplica.

## Criterio de éxito (recordatorio)

Mejora o igualdad en `macro_dice_fg` / `macro_iou_fg` en test (en condiciones comparables) sin regresión grave en **T9–T11** (ver plan).

## Decisión cerrada (`_cuda`, 2026-05-14)

**No adoptar** el letterbox multiclase en el procedimiento base con el run registrado: `macro_dice_fg` e `macro_iou_fg` en test **bajan** frente a `training_runs_cascade_v3/`. Detalle en `notebooks/v3_local/mejoras/RESULTADOS_Y_DECISIONES_GENERAL.md` (Fase 1) y en el notebook `_cuda` (análisis final).

**Siguiente fase:** `../train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase2_pesos_clases_t7_t12/README.md`.

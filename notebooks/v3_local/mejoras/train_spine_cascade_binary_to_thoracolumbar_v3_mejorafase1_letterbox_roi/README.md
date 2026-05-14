# Fase 1 — Letterbox en ROI multiclase (`mejorafase1_letterbox_roi`)

## Objetivo

Reducir la **deformación por estiramiento** al pasar el recorte de columna a la resolución fija del modelo multiclase, usando **letterbox** (escala uniforme + padding centrado). La máscara se rellena con `IGNORE_INDEX` (255) para que esos píxeles no entren en la `CrossEntropyLoss`.

## Notebooks (dos variantes)

| Archivo | Cuándo usarlo |
|---------|----------------|
| `train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_letterbox_roi_cpu.ipynb` | PC **sin GPU** o CUDA no compatible con tu PyTorch: resolución y épocas reducidas para acortar tiempo. |
| `train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_letterbox_roi_cuda.ipynb` | GPU con PyTorch compatible: mismo orden de magnitud que el cascada V3 base (512×256, más épocas). |

Cada notebook incluye al inicio un **mapa de cambios** respecto al V3 base y marcas **`[FASE 1]`** en markdown y código donde se implementa el letterbox. Tras cada entrenamiento, **actualiza** las celdas de interpretación y conclusiones para que coincidan con los CSV generados (no dejes texto copiado del base si los números cambian).

La primera celda de código resuelve la raíz del repo con `_resolve_maia_project_root`: sube por los padres de `Path.cwd()`, y en **Google Colab** busca bajo `/content/drive/MyDrive/Colab Notebooks/` (p. ej. `.../Colab Notebooks/MAIA-PROYECTO`). Si el repo está en otra ruta, define antes `os.environ['MAIA_PROJECT_ROOT'] = r'/ruta/absoluta/al/repo'` o ejecuta `%cd /ruta/al/repo` y vuelve a correr la celda de imports.

**Regenerar ambos notebooks** (por si actualizas el V3 base):

```bash
python mejoras/scripts/build_fase1_letterbox_notebooks.py
```

(El script `build_fase1_letterbox_notebook.py` sin **s** delega al mismo flujo.)

## Salidas (no pisan el baseline)

| Variante | Métricas / historiales | Pesos sugeridos |
|----------|------------------------|-----------------|
| `_cpu` | `analysis_outputs_v3/training_runs_cascade_v3_fase1_letterbox_cpu/` | `models/binary_spine_cascade_fase1_letterbox_cpu_best.pt`, `models/thoracolumbar_core_cascade_fase1_letterbox_cpu_best.pt` |
| `_cuda` | `analysis_outputs_v3/training_runs_cascade_v3_fase1_letterbox_cuda/` | `models/binary_spine_cascade_fase1_letterbox_cuda_best.pt`, `models/thoracolumbar_core_cascade_fase1_letterbox_cuda_best.pt` |

La variante `_cpu` prioriza **tiempo**; las métricas no son comparables 1:1 al baseline 512×256 sin salvedad. La variante `_cuda` es la referencia para comparar con `training_runs_cascade_v3/`.

## Entradas (no mover; solo lectura)

Igual que el notebook cascada V3: manifiesto, índice dataset V3, JSON de etiquetas (ver README fase 0).

## Tras ejecutar

1. Rellenar **§7.1** y filas en **§7.2** en `mejoras/RESULTADOS_Y_DECISIONES_GENERAL.md` (una entrada por variante si corres ambas).
2. Filas en `mejoras/experiment_registry.csv` (`1_cpu` / `1_cuda` o equivalente).
3. Comparar la variante `_cuda` con baseline en `analysis_outputs_v3/training_runs_cascade_v3/` (plan §1.2 y §4.2); documentar la `_cpu` como apoyo si aplica.

## Criterio de éxito (recordatorio)

Mejora o igualdad en `macro_dice_fg` / `macro_iou_fg` en test (en condiciones comparables) sin regresión grave en **T9–T11** (ver plan).

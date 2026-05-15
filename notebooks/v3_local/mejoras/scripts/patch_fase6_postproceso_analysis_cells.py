# -*- coding: utf-8 -*-
"""Inserta celdas markdown de cierre Fase 6 (analisis) antes del Registro de ejecucion.

Ejecutar **solo despues** de haber corrido el `*_cuda.ipynb` (o el `_cpu`) y **solo** sobre el
notebook de **esa** variante: las metricas difieren entre `_cpu` y `_cuda`; no reutilizar cifras.
"""
from __future__ import annotations

import json
from pathlib import Path

from mejoras_cierre_notebook_common import patch_phase_analysis_before_registry


BLOCKQUOTE = """> **Fase 6 (`_cuda`, islas + `FASE6_VERTICAL_MEDIAN_K=0`):** lectura del **run ejecutado** frente al **mismo modelo sin post-proceso** en test, al **baseline** y a **Fase 4 `_cuda`** como referencia de macro alto; métricas en `training_runs_cascade_v3_fase6_postproc_cuda/`.
"""

ANALYSIS = """### Análisis e interpretación

#### Test multiclase — sin post vs con post Fase 6 (islas mínimas `FASE6_MIN_ISLAND_PIXELS=64`, sin mediana vertical)

| Métrica | Sin post (`thoracolumbar_core_test_metrics.csv`) | Con post (`…_test_metrics_fase6_post.csv`) | Δ |
|---------|---------------------------------------------------|---------------------------------------------|---|
| `macro_dice_fg` | **0,2294** | **0,2273** | **−0,0021** |
| `macro_iou_fg` | **0,1360** | **0,1349** | **−0,0011** |
| `pixel_accuracy` | 0,7498 | 0,7526 | +0,0028 |

**Binario (test):** sin cambio respecto al post multiclase — Dice **0,8710** (`binary_spine_test_metrics.csv`).

#### Dice por clase (test) — foco torácico (sin post → con post)

| Clase | Sin post | Con post | Δ |
|-------|----------|----------|---|
| T7 | 0,2121 | 0,2072 | −0,0049 |
| T8 | 0,2060 | 0,2082 | +0,0022 |
| T9 | 0,1128 | 0,1029 | −0,0099 |
| T10 | 0,0777 | 0,0695 | −0,0082 |
| T11 | 0,1783 | 0,1752 | −0,0031 |
| T12 | 0,0983 | 0,0931 | −0,0052 |
| L5 | 0,1925 | 0,1925 | ≈0 |

#### Lectura frente al plan (§3 Fase 6)

- **Macro FG / IoU FG:** el post de **islas** **no** mejora el objetivo principal; baja ligeramente el macro Dice/IoU frente al test sin post.
- **Torácico (5.5 / §4.2):** varias vértebras T7–T12 **empeoran** o apenas cambian; no hay “ganancia lumbar sin coste torácico”.
- **Mediana vertical (`K=3`):** **no** se ejecutó; con el resultado actual **no** era necesario para decidir (riesgo de más degradación sin hipótesis favorable).

**Decisión de fase:** **No adoptar** integrar este post-proceso (islas 64 px, `K=0`) en el pipeline vivo; mantener inferencia multiclase **sin** este paso.

"""

SUG = """### Sugerencias posteriores

1. **Fase 7** del plan (estimador de última vértebra visible / clipping): ver `notebooks/07_colab_train_last_visible_estimator_and_clip_thoracolumbar_explained.ipynb` y la carpeta `notebooks/v3_local/mejoras/train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase7_auxiliares_rango_lastvis/`.
2. Si se reabre Fase 6: probar **solo** otros umbrales de isla o post **por clase** acotado, siempre midiendo macro FG y T9–T12 antes/después.
3. Tras cada run, ejecutar la celda de **Registro de ejecución** al pie del notebook.
"""


def main() -> None:
    d = Path(__file__).resolve().parents[1] / "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase6_postproceso_ligero"
    fname = "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase6_postproceso_ligero_cuda.ipynb"
    p = d / fname
    nb = json.loads(p.read_text(encoding="utf-8"))
    if patch_phase_analysis_before_registry(
        nb, [BLOCKQUOTE, ANALYSIS, SUG], already_markers=("sin post vs con post Fase 6",)
    ):
        p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("patched", fname)
    else:
        print("skip (ya insertado)", fname)


if __name__ == "__main__":
    main()

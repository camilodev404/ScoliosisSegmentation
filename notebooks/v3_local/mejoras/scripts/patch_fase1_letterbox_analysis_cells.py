# -*- coding: utf-8 -*-
"""Inserta cierre Fase 1 letterbox (`_cuda`) antes del Registro."""
from __future__ import annotations

import json
from pathlib import Path

from mejoras_cierre_notebook_common import patch_phase_analysis_before_registry

BLOCKQUOTE = """> **Fase 1 (`_cuda`, letterbox ROI multiclase):** lectura del **run ejecutado** frente al **cascada V3 base**; métricas en `training_runs_cascade_v3_fase1_letterbox_cuda/`.
"""

ANALYSIS = """### Análisis e interpretación

Run **Fase 1 `_cuda`** (letterbox en crop multiclase) frente al **cascada V3 base** (`training_runs_cascade_v3/`). Cifras de **test**.

#### 1. Resumen cuantitativo (test)

| Métrica | Baseline cascada | Fase 1 `_cuda` | Comentario |
|--------|------------------|----------------|------------|
| Binario — Dice | 0,8713 | 0,8581 | Ligera baja |
| Binario — IoU | 0,7720 | 0,7514 | Coherente con Dice |
| Multiclase — `macro_dice_fg` | 0,1894 | 0,1807 | **−0,0087** |
| Multiclase — `macro_iou_fg` | 0,1126 | 0,1038 | **−0,0088** |

#### 2. ROI

**198/198** `pred_binary` en `thoracolumbar_core_binary_rois.csv`.

#### 3. Desglose por vértebra (test)

| Vértebra | Baseline — Dice | Fase 1 `_cuda` — Dice |
|----------|-----------------|------------------------|
| T9 | 0,0136 | **0,1035** |
| T10 | 0,0440 | 0,0279 |
| T11 | 0,0296 | **0,1079** |
| L5 | 0,2158 | 0,2330 |

#### 4. Validación

Mejor `val_macro_dice_fg` ≈ **0,2123** (época **24**) — por encima del baseline en val, pero test macro menor.

#### 5. Decisión (cierre de fase)

**No adoptar** letterbox multiclase en el procedimiento base con esta configuración. Ver `RESULTADOS_Y_DECISIONES_GENERAL.md` § Fase 1.

"""

SUG = """### Sugerencias posteriores

1. **Fase 2** — pesos CE T7–T12 según plan.

2. Si se reabre letterbox: probar solo en subset o con menos padding antes de descartar del todo.
"""


def main() -> None:
    p = (
        Path(__file__).resolve().parents[1]
        / "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_letterbox_roi"
        / "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_letterbox_roi_cuda.ipynb"
    )
    nb = json.loads(p.read_text(encoding="utf-8"))
    if patch_phase_analysis_before_registry(
        nb, [BLOCKQUOTE, ANALYSIS, SUG], already_markers=("Run **Fase 1 `_cuda`**",)
    ):
        p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("patched", p.name)
    else:
        print("skip (ya insertado)", p.name)


if __name__ == "__main__":
    main()

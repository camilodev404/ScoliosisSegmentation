# -*- coding: utf-8 -*-
"""Inserta cierre Fase 2 pesos T7–T12 (`_cuda`) antes del Registro."""
from __future__ import annotations

import json
from pathlib import Path

from mejoras_cierre_notebook_common import patch_phase_analysis_before_registry

BLOCKQUOTE = """> **Fase 2 (`_cuda`, pesos CE ×1,40 en T7–T12):** lectura del **run ejecutado** frente al **cascada V3 base**; métricas en `training_runs_cascade_v3_fase2_pesos_t7_t12_cuda/`.
"""

ANALYSIS = """### Análisis e interpretación

Run **Fase 2 `_cuda`** (multiplicador **1,40×** en CE para clases T7–T12) frente al **cascada V3 base**. Cifras de **test**.

#### 1. Resumen cuantitativo (test)

| Métrica | Baseline cascada | Fase 2 `_cuda` | Comentario |
|--------|------------------|----------------|------------|
| Binario — Dice | 0,8713 | 0,8767 | Ligera mejora |
| Binario — IoU | 0,7720 | 0,7805 | Coherente |
| Multiclase — `macro_dice_fg` | 0,1894 | 0,1896 | **+0,0002** (marginal) |
| Multiclase — `macro_iou_fg` | 0,1126 | 0,1104 | **−0,0022** |

#### 2. ROI

**198/198** `pred_binary`.

#### 3. Desglose T7–T12 y L5 (test — Dice)

| Clase | Baseline | Fase 2 `_cuda` |
|-------|----------|----------------|
| T7 | 0,074 | **0,161** |
| T10 | 0,044 | **0,114** |
| T12 | 0,091 | **0,128** |
| L5 | **0,216** | 0,147 |

#### 4. Decisión (cierre de fase)

**No adoptar** el multiplicador **1,40×** solo en CE: mejora local T7–T12 con coste en **L5** y macro IoU. Ver `RESULTADOS_Y_DECISIONES_GENERAL.md` § Fase 2.

"""

SUG = """### Sugerencias posteriores

1. **Fase 3** — LR multiclase + scheduler (siguiente palanca del plan).

2. Si se reabre Fase 2: multiplicador más bajo (1,15–1,25) o peso análogo en rama Dice.
"""


def main() -> None:
    p = (
        Path(__file__).resolve().parents[1]
        / "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase2_pesos_clases_t7_t12"
        / "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase2_pesos_clases_t7_t12_cuda.ipynb"
    )
    nb = json.loads(p.read_text(encoding="utf-8"))
    if patch_phase_analysis_before_registry(
        nb, [BLOCKQUOTE, ANALYSIS, SUG], already_markers=("Run **Fase 2 `_cuda`**",)
    ):
        p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("patched", p.name)
    else:
        print("skip (ya insertado)", p.name)


if __name__ == "__main__":
    main()

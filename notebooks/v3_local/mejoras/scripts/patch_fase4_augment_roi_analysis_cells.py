# -*- coding: utf-8 -*-
"""Inserta cierre Fase 4 (`_cuda`) antes del Registro. Ejecutar solo tras el run de esa variante."""
from __future__ import annotations

import json
from pathlib import Path

from mejoras_cierre_notebook_common import patch_phase_analysis_before_registry

BLOCKQUOTE = """> **Fase 4 (`_cuda`):** lectura del **run ejecutado** frente al **baseline cascada**, al **Fase 3 run B** (`…_fase3_scheduler_lr_cuda_repro_b/`) y a la ruta de salida de esta fase. Las tablas con cifras están en la celda siguiente (**Análisis e interpretación**).
"""

ANALYSIS = """### Análisis e interpretación

Run **Fase 4 `_cuda`** (augment ROI train) frente a **baseline** y **Fase 3 run B** (misma base de entrenamiento salvo augment). Métricas de **test**.

#### 1. Resumen cuantitativo (test)

| Métrica | Baseline | Fase 3 run B | Fase 4 `_cuda` | Comentario |
|--------|----------|--------------|----------------|------------|
| Binario — Dice | 0,8713 | 0,8710 | **0,8710** | Alineado con baseline y run B |
| Binario — IoU | 0,7720 | 0,7714 | **0,7714** | Coherente |
| Multiclase — `macro_dice_fg` | 0,1894 | 0,2345 | **0,2380** | **+0,0035** vs run B; **+0,0486** vs baseline |
| Multiclase — `macro_iou_fg` | 0,1126 | 0,1387 | **0,1415** | Mejora respecto a B y baseline |

#### 2. ROI

**198/198** filas con `roi_source = pred_binary` en `thoracolumbar_core_binary_rois.csv`.

#### 3. Desglose por vértebra (test)

| Vértebra | Fase 3 run B — Dice | Fase 4 `_cuda` — Dice |
|----------|---------------------|------------------------|
| T9 | 0,1734 | 0,1324 |
| T10 | 0,0274 | **0,0731** |
| T11 | 0,1430 | **0,1750** |
| L5 | 0,2103 | 0,1955 |

#### 4. Validación

Mejor `val_macro_dice_fg` ≈ **0,2325** en época **22**.

#### 5. Decisión (cierre de fase)

**Adoptar** — mantener augment ROI en train en el notebook vivo (`apply_roi_augment=True`). Detalle en `RESULTADOS_Y_DECISIONES_GENERAL.md` § Fase 4.

"""

SUG = """### Sugerencias posteriores

1. **Fase 5 (multiseed):** replicar con 2–3 semillas para consolidar la magnitud de la ganancia (plan §3).

2. **Estratificación Normal/Escoliosis** en test si el plan lo exige.

3. **Afinar augment:** si T9 sigue sensible, reducir `max_deg` o el rango de escala (solo en experimentos nuevos).

4. **Colab `03_…`:** portar augment + Fase 3 cuando se alinee la rama oficial.
"""


def main() -> None:
    p = (
        Path(__file__).resolve().parents[1]
        / "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase4_augmentacion_roi"
        / "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase4_augmentacion_roi_cuda.ipynb"
    )
    nb = json.loads(p.read_text(encoding="utf-8"))
    if patch_phase_analysis_before_registry(
        nb, [BLOCKQUOTE, ANALYSIS, SUG], already_markers=("Run **Fase 4 `_cuda`**",)
    ):
        p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("patched", p.name)
    else:
        print("skip (ya insertado)", p.name)


if __name__ == "__main__":
    main()

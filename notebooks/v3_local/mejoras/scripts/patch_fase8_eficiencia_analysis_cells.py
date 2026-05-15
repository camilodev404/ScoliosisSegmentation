# -*- coding: utf-8 -*-
"""Inserta celdas markdown de cierre Fase 8 (analisis) antes del Registro de ejecucion.

Ejecutar **solo despues** del run `*_cuda.ipynb` de **esta** variante.
"""
from __future__ import annotations

import json
from pathlib import Path

from mejoras_cierre_notebook_common import patch_phase_analysis_before_registry

BLOCKQUOTE = """> **Fase 8 (`_cuda`, eficiencia):** resolución **384×192** + `UNetSmall` `base=24` vs vivo **512×256** / `base=32`. Métricas en `training_runs_cascade_v3_fase8_eficiencia_cuda/`.
"""

ANALYSIS = """### Análisis e interpretación

#### Test multiclase — Fase 8 vs Fase 4 (vivo)

| Métrica | Fase 8 `_cuda` | Fase 4 `_cuda` | Δ vs F4 |
|---------|----------------|----------------|---------|
| `macro_dice_fg` | **0,2157** | **0,2380** | **−0,0223** |
| `macro_iou_fg` | **0,1319** | **0,1415** | **−0,0096** |
| `pixel_accuracy` | 0,7495 | 0,7502 | −0,0007 |
| Binario — Dice | **0,8906** | 0,8710 | +0,0196 |

**Mejor `val_macro_dice_fg`:** **0,2163** (época **20**). Fase 4: **0,2325** (ép. 22).

**Criterio plan (§3 Fase 8):** adoptar si |Δ `macro_dice_fg`| ≤ **0,01** → **no cumple** (brecha **0,0223** > 0,01).

#### Dice por clase (test)

| Clase | Fase 8 | Fase 4 | Δ |
|-------|--------|--------|---|
| T9 | 0,1946 | 0,1324 | +0,0622 |
| T10 | 0,1100 | 0,0731 | +0,0369 |
| T11 | 0,1618 | 0,1750 | −0,0131 |
| L5 | 0,2785 | 0,1955 | +0,0830 |

Subidas locales en T9/T10/L5 **no** compensan la caída del **macro** frente a Fase 4 (otras clases y equilibrio global empeoran).

#### Eficiencia

Configuración: ~44% menos píxeles por imagen y ~44% menos canales base en U-Net. Los tiempos `binary_elapsed_min` / `multiclass_elapsed_min` deben tomarse del log Colab de este run si se documenta ahorro cuantitativo; el criterio de adopción del plan prioriza **métricas**, no solo velocidad.

#### Lectura

- **Calidad:** el perfil compacto **no** mantiene el macro multiclase del pipeline adoptado (F3+F4 a 512×256 / `base=32`).
- **Binario:** Dice test **superior** al vivo en este run (+0,02); no justifica por sí solo sustituir resolución/arquitectura multiclase.
- **Trade-off:** aceptar coste del vivo para conservar **0,2380** macro test; explorar eficiencia solo en **inferencia** (export ONNX, batch) sin re-entrenar a menor resolución, salvo nuevo experimento acotado.

**Decisión de fase:** **No adoptar** — mantener notebook vivo en **512×256** y `UNetSmall` `base=32` (Fase 3 + 4).

"""

SUG = """### Sugerencias posteriores

1. **Producción:** conservar entrenamiento/inferencia al perfil Fase 4; usar Fase 8 solo como referencia de trade-off coste/calidad.
2. **Aislar factores:** re-ejecutar solo ↓ resolución **o** solo `base=24` si se quiere saber cuál penaliza más el macro.
3. **Inferencia:** medir latencia y VRAM del checkpoint Fase 4 sin re-entrenar a 384×192.
4. Ejecutar la celda de **Registro de ejecución** al pie del notebook.
"""


def main() -> None:
    d = Path(__file__).resolve().parents[1] / (
        "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase8_eficiencia"
    )
    fname = "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase8_eficiencia_cuda.ipynb"
    p = d / fname
    nb = json.loads(p.read_text(encoding="utf-8"))
    if patch_phase_analysis_before_registry(
        nb,
        [BLOCKQUOTE, ANALYSIS, SUG],
        already_markers=("fase8_eficiencia_cuda_2026-05-15",),
    ):
        p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("patched", fname)
    else:
        print("skip (ya insertado)", fname)


if __name__ == "__main__":
    main()

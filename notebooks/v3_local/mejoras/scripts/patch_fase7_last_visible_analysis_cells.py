# -*- coding: utf-8 -*-
"""Inserta celdas markdown de cierre Fase 7 (analisis) antes del Registro de ejecucion.

Ejecutar **solo despues** del run `*_cuda.ipynb` de **esta** variante (run completo cascada + §8).
"""
from __future__ import annotations

import json
from pathlib import Path

from mejoras_cierre_notebook_common import (
    patch_phase_analysis_before_registry,
    strip_cells_before_registry,
)

BLOCKQUOTE = """> **Fase 7 (`_cuda`, notebook autocontenido, run completo):** cascada V3 (secc. 1–7) + **last_visible / clipping** (secc. 8) cerrados. Métricas cascada en `training_runs_cascade_v3_fase7_lastvis_cuda/`; auxiliar en `training_runs_last_visible_fase7_cuda/`.
"""

ANALYSIS = """### Análisis e interpretación

#### A. Cascada (secciones 1–7) — test

| Métrica | Fase 7 `_cuda` | Fase 4 `_cuda` (vivo) | Fase 6 `_cuda` (sin post) | Δ vs Fase 4 |
|---------|----------------|------------------------|---------------------------|-------------|
| Binario — Dice | **0,8710** | 0,8710 | 0,8710 | ≈0 |
| Binario — IoU | **0,7714** | 0,7714 | 0,7714 | ≈0 |
| `macro_dice_fg` | **0,2258** | **0,2380** | 0,2294 | **−0,0122** |
| `macro_iou_fg` | **0,1341** | **0,1415** | 0,1360 | **−0,0074** |
| `pixel_accuracy` | 0,7470 | 0,7502 | 0,7498 | −0,0032 |

**Mejor `val_macro_dice_fg`:** **0,2247** (época **22**). Referencia Fase 4: **0,2325** (ép. 22).

**Dice por clase (test)** — foco plan:

| Clase | Fase 7 | Fase 4 | Fase 6 |
|-------|--------|--------|--------|
| T9 | 0,0872 | 0,1324 | 0,1128 |
| T10 | 0,0849 | 0,0731 | 0,0777 |
| T11 | 0,1771 | 0,1750 | 0,1783 |
| L5 | 0,1871 | 0,1955 | 0,1925 |

T10 **sube** ligeramente vs Fase 4; **T9 y L5 bajan**; macro global **no** supera Fase 4.

#### B. Last visible (sección 8) — test (`last_visible_summary.csv`, n=40)

| Métrica | Valor |
|---------|-------|
| Exactitud índice última vértebra | **0,25** |
| Within-1 | **0,425** |
| MAE (índices) | **2,025** |
| Tasa sobrepredicción (rango) | **0,525** |
| Tasa subpredicción | 0,225 |

El estimador **no** alcanza utilidad clínica operativa (exactitud baja; MAE alto). Val/ train: within-1 ~0,69–0,84, pero **test** queda muy por debajo → generalización insuficiente.

#### C. Clipping multiclase — test (`last_visible_clipping_metric_comparison.csv`)

| Modo | `macro_dice_fg` | `macro_iou_fg` | Nota |
|------|-----------------|----------------|------|
| `raw` (sin clip) | 0,2216 | 0,1315 | Predicción cascada de este run |
| `last_pred_clip` | **0,2295** | 0,1367 | Clip con índice estimado |
| `oracle_clip` | 0,2491 | 0,1508 | Techo con GT last_visible |
| `prev_range_clip` | 0,2827 | 0,1737 | Referencia histórica (CSV legacy); **no** comparable a adopción |

**Efecto conteo vértebras** (`last_visible_presence_summary.csv`): media vértebras extra **2,48 → 1,35** con `last_pred_clip`, pero faltantes **0,65 → 1,38** (9/40 muestras empeoran missing). Mejora de macro **+0,0079** vs `raw` del mismo run, pero **sigue por debajo** de Fase 4 sin clipping (0,2380).

#### Lectura frente al plan (§3 Fase 7)

1. **Cascada autocontenida** reproduce binario alineado al vivo pero **no** iguala ni supera Fase 4 en `macro_dice_fg` test.
2. **Last visible:** métricas de test insuficientes; no justifica integrar el auxiliar en el pipeline.
3. **Clipping `last_pred_clip`:** mejora marginal local frente a `raw` del mismo run; **no** compensa la brecha vs Fase 4 ni el coste en vértebras faltantes.

**Decisión de fase:** **No adoptar** — mantener pipeline vivo **Fase 3 + Fase 4**; **no** sustituir por checkpoints fase7 ni integrar `last_visible` / clipping en producción con este run.

"""

SUG = """### Sugerencias posteriores

1. **Fase 8** (eficiencia / inferencia) según `PLAN_ACCION_AJUSTES_MODELOS.md` §3.
2. Si se reabre last_visible: más datos, features o arquitectura; objetivo mínimo **within-1 test > 0,7** antes de reevaluar clipping.
3. Explorar clipping **solo** si el estimador mejora; `oracle_clip` muestra techo (~0,249) pero no es desplegable.
4. Tras cada run, ejecutar la celda de **Registro de ejecución** al pie del notebook.
"""


def _is_fase7_stale_analysis(text: str) -> bool:
    if "run incompleto" in text or "no cerrado" in text and "USE_AMP" in text:
        return True
    if "Decisión de fase (este run)" in text and "No evaluado" in text:
        return True
    if "fase7_lastvis_cuda_completo_2026-05-15" in text:
        return True
    return False


def _strip_all_closing_markdown_before_registry(nb: dict) -> None:
    from mejoras_cierre_notebook_common import find_registry_markdown_index, _is_inherited_base_analysis

    reg = find_registry_markdown_index(nb)
    if reg is None:
        return
    keys = (
        "run V3 ya ejecutado",
        "Textos heredados del notebook base",
        "Sugerencias para mejorar este notebook",
        "USE_AMP",
        "run incompleto",
        "fase7_lastvis_cuda_completo",
        "Cascada (secciones 1",
        "Fase 7 (`_cuda`",
        "### Sugerencias posteriores",
        "Decisión de fase",
    )
    to_del: list[int] = []
    for i in range(reg):
        if nb["cells"][i].get("cell_type") != "markdown":
            continue
        t = "".join(nb["cells"][i].get("source", []))
        if _is_inherited_base_analysis(t) or _is_fase7_stale_analysis(t) or any(k in t for k in keys):
            to_del.append(i)
    for i in sorted(to_del, reverse=True):
        del nb["cells"][i]


def patch_fase7_notebook(nb: dict) -> bool:
    joined = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    if "fase7_lastvis_cuda_completo_2026-05-15" in joined and "run completo" in joined:
        return False
    _strip_all_closing_markdown_before_registry(nb)
    return patch_phase_analysis_before_registry(
        nb,
        [BLOCKQUOTE, ANALYSIS, SUG],
        already_markers=("fase7_lastvis_cuda_completo_2026-05-15",),
    )


def main() -> None:
    d = Path(__file__).resolve().parents[1] / (
        "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase7_auxiliares_rango_lastvis"
    )
    fname = (
        "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase7_auxiliares_rango_lastvis_cuda.ipynb"
    )
    p = d / fname
    nb = json.loads(p.read_text(encoding="utf-8"))
    if patch_fase7_notebook(nb):
        p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("patched", fname)
    else:
        print("skip (ya insertado)", fname)


if __name__ == "__main__":
    main()

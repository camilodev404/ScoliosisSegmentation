# -*- coding: utf-8 -*-
"""Inserta celdas markdown de cierre Fase 5 (analisis) antes del Registro de ejecucion.

Ejecutar **solo despues** del run de **cada** semilla y **solo** sobre el notebook de esa semilla:
las tablas no deben copiarse entre `seed42` / `seed1337` / `seed4242` ni entre `_cpu`/`_cuda`.
"""
from __future__ import annotations

import json
from pathlib import Path

from mejoras_cierre_notebook_common import patch_phase_analysis_before_registry


BLOCKQUOTE = """> **Fase 5 (multiseed `_cuda`):** lectura del **run de esta variante** frente al **baseline cascada**, al **Fase 4 `_cuda`** (`training_runs_cascade_v3_fase4_augment_roi_cuda/`), a las **otras semillas** de esta misma fase y a la carpeta `OUTPUT_DIR` de este notebook.
"""

ANALYSIS_COMMON = """### Análisis e interpretación

#### Tabla comparativa — tres semillas Fase 5 (test)

| Semilla | `macro_dice_fg` | `macro_iou_fg` | Bin Dice | Bin IoU | Mejor `val_macro_dice_fg` (época) |
|---------|-----------------|----------------|----------|---------|-----------------------------------|
| 42 | 0,2294 | 0,1365 | 0,8710 | 0,7714 | 0,2226 (ép. 22) |
| 1337 | 0,2457 | 0,1503 | 0,8943 | 0,8088 | 0,2609 (ép. 19) |
| 4242 | 0,1737 | 0,1035 | 0,8582 | 0,7516 | 0,1685 (ép. 24) |

**Referencias:** baseline `macro_dice_fg` **0,1894** (`training_runs_cascade_v3/`); Fase 4 `_cuda` **0,2380** (`training_runs_cascade_v3_fase4_augment_roi_cuda/`).

#### Dice por clase (test): T9, T10, T11, L5

| Semilla | T9 | T10 | T11 | L5 |
|---------|-----|------|------|-----|
| 42 | 0,0935 | 0,0831 | 0,1977 | 0,1807 |
| 1337 | 0,1480 | 0,1261 | 0,0800 | 0,4178 |
| 4242 | 0,1283 | **0,0000** | 0,1653 | 0,3326 |

**ROI:** en los tres runs, `thoracolumbar_core_binary_rois.csv` contiene **198** filas con `roi_source=pred_binary`.

#### Lectura frente al plan (§3 Fase 5)

- **Replicación:** no estable entre semillas: **1337** supera a Fase 4 en macro FG (**+0,0077**); **42** queda por debajo (**−0,0086** vs 0,2380); **4242** cae por debajo del baseline en macro y exhibe **T10 con Dice 0** en test.
- **Binario:** 1337 mejora frente al baseline; 42 se alinea con el baseline histórico; 4242 muestra ligera baja de Dice.
- **Decisión de fase:** **«Prometedor; no consolidado»** — no se añade un cambio de código al vivo (ya integra Fase 3 + Fase 4); se documenta **alta varianza** del macro multiclase ante la semilla bajo el mismo split y augment.

"""

TAIL = {
    42: "**Esta variante (`SEED=42`):** macro test **0,2294** (control habitual); respecto al run cerrado Fase 4 `_cuda` (0,2380) la diferencia puede deberse a variación estadística y/o pequeñas diferencias de entorno; no invalida por sí sola el pipeline adoptado.\n",
    1337: "**Esta variante (`SEED=1337`):** mejor macro FG de la tanda (**0,2457**); el binario también sube (Dice **0,8943**). Sirve como evidencia de que **existen** semillas favorables, pero no bastan sin replicación estable.\n",
    4242: "**Esta variante (`SEED=4242`):** peor caso claro — macro por debajo del baseline y **T10** colapsada en test; alerta para futuros ajustes (augment, pesos o post-proceso) si se prioriza robustez en torácico medio.\n",
}

SUG = """### Sugerencias posteriores

1. **Fase 6** (`train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase6_postproceso_ligero/`): **cerrada** — post islas (`K=0`) **no** mejoró `macro_dice_fg` en test; decisión **No adoptar** (ver `RESULTADOS_Y_DECISIONES_GENERAL.md`). **Fase 7:** estimador `last_visible` / clipping — `notebooks/07_colab_train_last_visible_estimator_and_clip_thoracolumbar_explained.ipynb` y carpeta `…_mejorafase7_auxiliares_rango_lastvis/`.
2. Opcional: ampliar a **más semillas** o reportar media ± desviación estándar sobre ≥5 runs para informes finales.
3. Tras cada run, confirmar que existan `thoracolumbar_core_test_metrics.csv` y el bloque de **Registro de ejecución** al pie del notebook.
"""


def main() -> None:
    d = Path(__file__).resolve().parents[1] / "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase5_multiseed"
    for seed, fname in [
        (42, "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase5_multiseed_cuda_seed42.ipynb"),
        (1337, "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase5_multiseed_cuda_seed1337.ipynb"),
        (4242, "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase5_multiseed_cuda_seed4242.ipynb"),
    ]:
        p = d / fname
        nb = json.loads(p.read_text(encoding="utf-8"))
        block = ANALYSIS_COMMON + TAIL[seed]
        if patch_phase_analysis_before_registry(
            nb,
            [BLOCKQUOTE, block, SUG],
            already_markers=("#### Tabla comparativa — tres semillas Fase 5",),
        ):
            p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
            print("patched", fname)
        else:
            print("skip (ya insertado)", fname)


if __name__ == "__main__":
    main()

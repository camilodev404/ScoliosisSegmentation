# -*- coding: utf-8 -*-
"""Actualiza el bloque 'Sugerencias posteriores' en los tres notebooks Fase 5 multiseed (texto post cierre Fase 6)."""
from __future__ import annotations

import json
from pathlib import Path

OLD = (
    "1. **Fase 6** del plan (post-proceso ligero anatómico) o experimentos acotados si se quiere acotar colapsos tipo T10 bajo malas semillas.\n"
)
NEW = (
    "1. **Fase 6** (`train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase6_postproceso_ligero/`): **cerrada** — post islas (`K=0`) **no** mejoró `macro_dice_fg` en test; decisión **No adoptar** (ver `RESULTADOS_Y_DECISIONES_GENERAL.md`). **Fase 7:** estimador `last_visible` / clipping — `notebooks/07_colab_train_last_visible_estimator_and_clip_thoracolumbar_explained.ipynb` y carpeta `…_mejorafase7_auxiliares_rango_lastvis/`.\n"
)


def main() -> None:
    d = Path(__file__).resolve().parents[1] / "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase5_multiseed"
    for fname in [
        "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase5_multiseed_cuda_seed42.ipynb",
        "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase5_multiseed_cuda_seed1337.ipynb",
        "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase5_multiseed_cuda_seed4242.ipynb",
    ]:
        p = d / fname
        nb = json.loads(p.read_text(encoding="utf-8"))
        changed = False
        for c in nb["cells"]:
            if c.get("cell_type") != "markdown":
                continue
            s = "".join(c.get("source", []))
            if "### Sugerencias posteriores" not in s or OLD not in s:
                continue
            s2 = s.replace(OLD, NEW, 1)
            if s2 != s:
                parts = s2.splitlines(keepends=True)
                c["source"] = parts if parts else [""]
                changed = True
                break
        if not changed:
            print("skip or already updated", fname)
            continue
        p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("updated", fname)


if __name__ == "__main__":
    main()

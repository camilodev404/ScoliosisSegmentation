# -*- coding: utf-8 -*-
"""Añade Registro + análisis Fase 1/2 `_cuda` si el notebook se generó sin bloque final."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cascade_v3_mejora_notebook_common import append_execution_registry_cells

HERE = Path(__file__).resolve().parent
MEJORAS = HERE.parent

NOTEBOOKS = [
    MEJORAS
    / "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_letterbox_roi"
    / "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_letterbox_roi_cuda.ipynb",
    MEJORAS
    / "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase2_pesos_clases_t7_t12"
    / "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase2_pesos_clases_t7_t12_cuda.ipynb",
]

PATCHES = [
    "patch_fase1_letterbox_analysis_cells.py",
    "patch_fase2_pesos_t7_t12_analysis_cells.py",
]


def ensure_registry(p: Path) -> None:
    nb = json.loads(p.read_text(encoding="utf-8"))
    if any("Registro de ejecución del notebook" in "".join(c.get("source", [])) for c in nb["cells"]):
        return
    append_execution_registry_cells(nb)
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("registry added:", p.name)


def main() -> None:
    for p in NOTEBOOKS:
        ensure_registry(p)
    for name in PATCHES:
        subprocess.run([sys.executable, str(HERE / name)], check=True, cwd=str(HERE))


if __name__ == "__main__":
    main()

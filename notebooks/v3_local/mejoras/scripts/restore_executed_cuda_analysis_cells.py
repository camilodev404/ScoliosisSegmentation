# -*- coding: utf-8 -*-
"""
Reinserta en los notebooks `_cuda` ya ejecutados los bloques de analisis documentados
(cifras alineadas con RESULTADOS_Y_DECISIONES_GENERAL.md).

Uso (raiz del repo):
  python notebooks/v3_local/mejoras/scripts/restore_executed_cuda_analysis_cells.py

No usar en notebooks `_cpu` sin ejecutar ni en Fase 7 sin run.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "fix_fase1_fase2_cuda_registry_and_analysis.py",
    "patch_fase1_letterbox_analysis_cells.py",
    "patch_fase2_pesos_t7_t12_analysis_cells.py",
    "patch_fase4_augment_roi_analysis_cells.py",
    "patch_fase5_multiseed_analysis_cells.py",
    "patch_fase6_postproceso_analysis_cells.py",
]


def main() -> None:
    here = Path(__file__).resolve().parent
    for name in SCRIPTS:
        p = here / name
        if not p.exists():
            print("skip (no existe)", name)
            continue
        print("---", name)
        subprocess.run([sys.executable, str(p)], check=True, cwd=str(here))


if __name__ == "__main__":
    main()

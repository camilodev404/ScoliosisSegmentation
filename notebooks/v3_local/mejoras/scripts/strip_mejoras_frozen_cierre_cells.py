# -*- coding: utf-8 -*-
"""
DEPRECADO ? no ejecutar salvo regeneracion controlada.

Este script eliminaba celdas de analisis documentado; contradice el flujo acordado:
- Tras `build_fase*`: solo plantilla **vacia** (`prepare_cierre_vacio_before_registry` en builds).
- Tras ejecutar `_cuda` y analizar: usar `patch_fase*_analysis_cells.py` o
  `restore_executed_cuda_analysis_cells.py`.

Para notebooks nuevos desde build, el analisis heredado del base se quita en
`mejoras_cierre_notebook_common.prepare_cierre_vacio_before_registry`.
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    print(__doc__)
    sys.exit(1)

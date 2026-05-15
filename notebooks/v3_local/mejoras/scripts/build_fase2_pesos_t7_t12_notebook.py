# -*- coding: utf-8 -*-
"""
Punto de entrada legado: genera **ambos** notebooks Fase 2 (`_cpu` y `_cuda`).

El flujo oficial está en `build_fase2_pesos_t7_t12_notebooks.py`.
"""
from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    sibling = Path(__file__).with_name("build_fase2_pesos_t7_t12_notebooks.py")
    runpy.run_path(str(sibling), run_name="__main__")


if __name__ == "__main__":
    main()

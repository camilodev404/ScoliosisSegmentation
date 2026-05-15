# -*- coding: utf-8 -*-
"""
Punto de entrada legado: genera **ambos** notebooks Fase 3 (`_cpu` y `_cuda`).

El flujo oficial está en `build_fase3_scheduler_lr_notebooks.py`.
"""
from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    sibling = Path(__file__).with_name("build_fase3_scheduler_lr_notebooks.py")
    runpy.run_path(str(sibling), run_name="__main__")


if __name__ == "__main__":
    main()

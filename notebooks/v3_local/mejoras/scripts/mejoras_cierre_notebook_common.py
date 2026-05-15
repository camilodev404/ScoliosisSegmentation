# -*- coding: utf-8 -*-
"""
Cierre de notebooks de mejoras: plantilla vacía al generar; inserción de análisis tras ejecutar.

Convención:
- Tras `build_fase*`: sección **Análisis e interpretación** vacía (lista para diligenciar).
- Tras ejecutar y analizar `_cuda` (u otra variante): `patch_fase*_analysis_cells.py` con cifras de **ese** run.
- No copiar tablas entre `_cpu` / `_cuda` ni entre semillas.
"""
from __future__ import annotations

import uuid
from typing import Any


CIERRE_VACIO_MD = """### Análisis e interpretación

*(Completar **después** de ejecutar este notebook y revisar los CSV en `OUTPUT_DIR`. Incluir tablas de test, lectura frente al baseline / fases previas y decisión adoptar / no adoptar. **No** copiar cifras de otra variante `_cpu` / `_cuda` ni de otra semilla.)*

### Sugerencias posteriores

*(Opcional — próximos experimentos o integración al notebook base.)*
"""


def lines_from_str(s: str) -> list[str]:
    parts = s.splitlines(keepends=True)
    return parts if parts else [""]


def new_markdown_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": lines_from_str(text),
    }


def _cell_text(c: dict) -> str:
    return "".join(c.get("source", []))


def find_registry_markdown_index(nb: dict) -> int | None:
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") == "markdown" and "Registro de ejecución del notebook" in _cell_text(c):
            return i
    return None


def _is_vacio_or_placeholder(text: str) -> bool:
    if "Cierre del run (analisis en el notebook)" in text:
        return True
    if "### Análisis e interpretación" in text and "*(Completar **después**" in text:
        return True
    return False


def _is_inherited_base_analysis(text: str) -> bool:
    """Tabla genérica plano vs cascada heredada del notebook base (no es cierre de fase)."""
    if "### Análisis e interpretación" not in text:
        return False
    if "**run V3 ya ejecutado**" in text and "training_runs_v3" in text:
        return True
    if "| Métrica | Plano | Cascada |" in text:
        return True
    return False


def strip_cells_before_registry(nb: dict, predicate) -> int:
    """Elimina celdas markdown contiguas antes del Registro que cumplen predicate. Retorna n eliminadas."""
    reg_i = find_registry_markdown_index(nb)
    if reg_i is None:
        return 0
    to_delete: list[int] = []
    j = reg_i - 1
    while j >= 0 and nb["cells"][j].get("cell_type") == "markdown":
        if predicate(_cell_text(nb["cells"][j])):
            to_delete.append(j)
            j -= 1
            continue
        break
    for idx in sorted(to_delete, reverse=True):
        del nb["cells"][idx]
    return len(to_delete)


def prepare_cierre_vacio_before_registry(nb: dict) -> None:
    """Quita análisis heredado del base / placeholder antiguo; deja plantilla vacía antes del Registro."""
    strip_cells_before_registry(nb, _is_vacio_or_placeholder)
    strip_cells_before_registry(nb, _is_inherited_base_analysis)
    reg_i = find_registry_markdown_index(nb)
    if reg_i is None:
        return
    joined = "\n".join(_cell_text(c) for c in nb["cells"])
    if "*(Completar **después**" in joined:
        return
    nb["cells"].insert(reg_i, new_markdown_cell(CIERRE_VACIO_MD))


def insert_markdown_cells_before_registry(nb: dict, texts: list[str]) -> int | None:
    """Inserta celdas markdown (en orden) justo antes del Registro. Retorna índice de inserción."""
    reg_i = find_registry_markdown_index(nb)
    if reg_i is None:
        return None
    new_cells = [new_markdown_cell(t) for t in texts]
    nb["cells"] = nb["cells"][:reg_i] + new_cells + nb["cells"][reg_i:]
    return reg_i


def patch_phase_analysis_before_registry(nb: dict, texts: list[str], *, already_markers: tuple[str, ...]) -> bool:
    """
    Inserta bloques de cierre de fase antes del Registro.
    Si `already_markers` ya están en el notebook, no hace nada (idempotente).
  """
    joined = "\n".join(_cell_text(c) for c in nb["cells"])
    if any(m in joined for m in already_markers):
        return False
    strip_cells_before_registry(nb, _is_vacio_or_placeholder)
    strip_cells_before_registry(nb, _is_inherited_base_analysis)
    if insert_markdown_cells_before_registry(nb, texts) is None:
        return False
    return True

# -*- coding: utf-8 -*-
"""Aplica correcciones runtime Fase 7 en notebooks ya generados (sin regenerar todo).

  python notebooks/v3_local/mejoras/scripts/patch_fase7_runtime_fixes.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from cascade_v3_mejora_notebook_common import find_config_cell_index, lines_from_str, new_code_cell, new_markdown_cell

DST = Path(__file__).resolve().parents[1] / (
    "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase7_auxiliares_rango_lastvis"
)

FASE7_RESUME_MD = """### [FASE 7] Reanudar tras parches del notebook

Si **ya** corriste cascada + last_visible y solo actualizaste el `.ipynb`:

1. Ejecuta la celda **config** (imports + aliases).
2. Continúa desde la celda que falló (clipping / métricas / export).

**No** hace falta repetir entrenamiento salvo **Runtime → Reiniciar** el kernel.
"""

FASE7_BOOTSTRAP = """# --- [FASE 7] Aliases idempotentes (ejecutar si solo re-corres §8+) ---
TARGET_SUBSET = MULTICLASS_SUBSET
"""

_ALIAS_LINE = "TARGET_SUBSET = MULTICLASS_SUBSET  # alias notebook 07\n"


def _dedupe_config_alias(s: str) -> str:
    s = s.replace("MULTICLASS_SUBSET = MULTICLASS_SUBSET  # alias notebook 07\n", _ALIAS_LINE)
    while s.count(_ALIAS_LINE) > 1:
        s = s.replace(_ALIAS_LINE, "", 1)
    return s


def _patch_config(s: str) -> str:
    if "import copy\n" not in s:
        s = s.replace("import json\n", "import copy\nimport json\n", 1)
    if "USE_AMP" not in s:
        s = s.replace(
            "USE_MULTICLASS_TTA = True\n",
            "USE_MULTICLASS_TTA = True\nUSE_AMP = DEVICE.type == \"cuda\"\n",
            1,
        )
    if "PREV_RANGE_TEST_PATH" not in s:
        block = (
            "\nPREV_RANGE_TEST_PATH = (\n"
            "    ROOT / 'reports' / 'analysis_outputs' / 'visible_range_estimator_thoracolumbar_explained'\n"
            "    / 'visible_range_test_predictions.csv'\n"
            ")\n"
        )
        if "LAST_VISIBLE_MODEL_PATH" in s:
            out: list[str] = []
            for line in s.splitlines(keepends=True):
                out.append(line)
                if line.strip().startswith("LAST_VISIBLE_MODEL_PATH ="):
                    out.append(block)
            s = "".join(out)
    if "TARGET_SUBSET = MULTICLASS_SUBSET" not in s and "MULTICLASS_SUBSET" in s:
        s = re.sub(
            r"(MULTICLASS_SUBSET = ['\"]core['\"][^\n]*\n)",
            r"\1TARGET_SUBSET = MULTICLASS_SUBSET  # alias notebook 07\n",
            s,
            count=1,
        )
    return _dedupe_config_alias(s)


def _patch_code_cells(s: str) -> str:
    s = s.replace(
        "scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)",
        "scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)",
    )
    s = s.replace(
        "with torch.cuda.amp.autocast(enabled=USE_AMP):",
        "with torch.amp.autocast('cuda', enabled=USE_AMP):",
    )
    # Baseline usa MULTICLASS_SUBSET; el 07 usa TARGET_SUBSET (no tocar líneas alias)
    s = re.sub(
        r"(?<!MULTICLASS_)TARGET_SUBSET",
        "MULTICLASS_SUBSET",
        s,
    )
    s = s.replace(
        "MULTICLASS_SUBSET = MULTICLASS_SUBSET  # alias notebook 07\n",
        "TARGET_SUBSET = MULTICLASS_SUBSET  # alias notebook 07\n",
    )
    return s


def _insert_bootstrap_before_section8(nb: dict) -> bool:
    if any("Aliases idempotentes" in "".join(c.get("source", [])) for c in nb["cells"]):
        return False
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "markdown":
            continue
        t = "".join(c.get("source", []))
        if "Seccion 8. Fase 7" in t:
            nb["cells"].insert(i, new_code_cell(FASE7_BOOTSTRAP.strip()))
            nb["cells"].insert(i, new_markdown_cell(FASE7_RESUME_MD))
            return True
    return False


def patch_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    ic = find_config_cell_index(nb)
    nb["cells"][ic]["source"] = lines_from_str(_patch_config("".join(nb["cells"][ic]["source"])))
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        s2 = _patch_code_cells(src)
        if s2 != src:
            c["source"] = lines_from_str(s2)
    _insert_bootstrap_before_section8(nb)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("patched", path.name)


def main() -> None:
    for name in (
        "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase7_auxiliares_rango_lastvis_cuda.ipynb",
        "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase7_auxiliares_rango_lastvis_cpu.ipynb",
    ):
        p = DST / name
        if p.exists():
            patch_notebook(p)


if __name__ == "__main__":
    main()

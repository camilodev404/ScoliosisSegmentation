# -*- coding: utf-8 -*-
"""
Genera notebooks Fase 8 — eficiencia (menor resolución + U-Net compacta).

Parte del cascada V3 vivo (Fase 3 + 4). Cambios experimentales:
- `IMG_SIZE_*` reducido (cuda: 384×192 vs vivo 512×256).
- `UNET_BASE_CHANNELS = 24` (vivo: 32 por defecto en `UNetSmall`).

Desde la raíz del repo:
  python notebooks/v3_local/mejoras/scripts/build_fase8_eficiencia_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

from cascade_v3_mejora_notebook_common import (
    COLAB_CUDA_PROJECT_ROOT,
    append_execution_registry_cells,
    clear_code_cell_outputs,
    find_config_cell_index,
    insert_markdown_before_read_gray,
    lines_from_str,
    patch_config_cell_for_training_variant,
    patch_markdown_antes_de_ejecutar_output_dir,
    prepend_interpretacion_warning,
    prepend_mapa_and_optional_colab_cuda_cells,
    replace_cascade_checkpoint_paths,
)

SRC = Path(__file__).resolve().parents[2] / "train_spine_cascade_binary_to_thoracolumbar_v3.ipynb"
DST_DIR = Path(__file__).resolve().parents[1] / (
    "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase8_eficiencia"
)

VARIANTS: dict[str, dict[str, str]] = {
    "cpu": {
        "label": "CPU (perfil liviano)",
        "output_dir": "training_runs_cascade_v3_fase8_eficiencia_cpu",
        "binary_pt": "binary_spine_cascade_fase8_eficiencia_cpu_best.pt",
        "multiclass_pt": "thoracolumbar_{MULTICLASS_SUBSET}_cascade_fase8_eficiencia_cpu_best.pt",
        "img_binary": "(256, 128)",
        "img_mc": "(256, 128)",
        "unet_base": "24",
        "batch": "2",
        "binary_epochs": "6",
        "multiclass_epochs": "8",
        "nota": (
            "Exploración local: 256×128 y U-Net base=24. Métricas no comparables 1:1 con `_cuda`."
        ),
    },
    "cuda": {
        "label": "CUDA (perfil completo — eficiencia)",
        "output_dir": "training_runs_cascade_v3_fase8_eficiencia_cuda",
        "binary_pt": "binary_spine_cascade_fase8_eficiencia_cuda_best.pt",
        "multiclass_pt": "thoracolumbar_{MULTICLASS_SUBSET}_cascade_fase8_eficiencia_cuda_best.pt",
        "img_binary": "(384, 192)",
        "img_mc": "(384, 192)",
        "unet_base": "24",
        "batch": "4",
        "binary_epochs": "14",
        "multiclass_epochs": "24",
        "nota": (
            "Mismas épocas que el vivo; resolución **384×192** (~44% menos píxeles que 512×256) "
            "y `UNetSmall(..., base=24)` (~44% menos canales base que 32). Criterio adopción: "
            "Δ `macro_dice_fg` test ≤ 0,01 vs Fase 4 `_cuda` (plan §3 Fase 8)."
        ),
    },
}


def patch_unet_base_channels(nb: dict, cfg: dict[str, str]) -> None:
    ic = find_config_cell_index(nb)
    s = "".join(nb["cells"][ic]["source"])
    line = f"UNET_BASE_CHANNELS = {cfg['unet_base']}  # Fase 8 (notebook vivo: 32)\n"
    if "UNET_BASE_CHANNELS" not in s:
        anchor = "IMG_SIZE_MULTICLASS = "
        if anchor not in s:
            raise RuntimeError("IMG_SIZE_MULTICLASS no encontrado en config")
        s = s.replace(anchor, line + anchor, 1)
        nb["cells"][ic]["source"] = lines_from_str(s)

    replacements = (
        ("UNetSmall(in_channels=1, out_channels=1)", "UNetSmall(in_channels=1, out_channels=1, base=UNET_BASE_CHANNELS)"),
        (
            "UNetSmall(in_channels=1, out_channels=num_classes)",
            "UNetSmall(in_channels=1, out_channels=num_classes, base=UNET_BASE_CHANNELS)",
        ),
    )
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "base=UNET_BASE_CHANNELS" in src:
            continue
        s2 = src
        for old, new in replacements:
            s2 = s2.replace(old, new)
        if s2 != src:
            cell["source"] = lines_from_str(s2)


def build_notebook(variant_key: str) -> Path:
    cfg = VARIANTS[variant_key]
    nb = json.loads(SRC.read_text(encoding="utf-8"))

    mapa_md = f"""# Fase 8 — Eficiencia (`_{variant_key}`)

## Finalidad

Según el plan (`PLAN_ACCION_AJUSTES_MODELOS.md` §3 Fase 8): reducir coste de entrenamiento e inferencia con **menor resolución** y **U-Net más compacta**, manteniendo el pipeline vivo (Fase 3 scheduler + Fase 4 augment ROI).

## Mapa de cambios (vs `train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` vivo)

| Ubicación | Qué cambia |
|-----------|-------------|
| Config | `IMG_SIZE_BINARY` / `IMG_SIZE_MULTICLASS` → **{cfg["img_binary"]}**; `UNET_BASE_CHANNELS = {cfg["unet_base"]}` (vivo: 32). |
| Modelos | `UNetSmall(..., base=UNET_BASE_CHANNELS)` en binario y multiclase. |
| Entrenamiento | Misma lógica F3+F4; `OUTPUT_DIR` y checkpoints propios de Fase 8. |

## Variante: **{cfg["label"]}**

{cfg["nota"]}

**Referencia métrica (vivo):** Fase 4 `_cuda` — `macro_dice_fg` test **0,2380** (`training_runs_cascade_v3_fase4_augment_roi_cuda/`).

---

"""
    if variant_key == "cuda":
        mapa_md += (
            "## Colab — raíz por defecto (`_cuda`)\n\n"
            f"`{COLAB_CUDA_PROJECT_ROOT.as_posix()}`\n\n---\n\n"
        )
    prepend_mapa_and_optional_colab_cuda_cells(nb, variant_key, mapa_md)

    insert_markdown_before_read_gray(
        nb,
        lines_from_str(
            "### [FASE 8 — EFICIENCIA] Referencia\n\n"
            f"- Resolución: `IMG_SIZE_BINARY` / `IMG_SIZE_MULTICLASS` = **{cfg['img_binary']}**.\n"
            f"- U-Net: `UNET_BASE_CHANNELS = {cfg['unet_base']}` (default vivo: 32).\n"
            "- Scheduler Fase 3 y augment Fase 4: sin cambio respecto al notebook vivo.\n"
            "- Tras el run: comparar `macro_dice_fg` test vs Fase 4; registrar tiempos `binary_elapsed_min` / `multiclass_elapsed_min` impresos al entrenar.\n"
        ),
    )

    patch_config_cell_for_training_variant(nb, cfg, variant_key, "FASE 8 eficiencia")
    patch_unet_base_channels(nb, cfg)
    replace_cascade_checkpoint_paths(nb, cfg)
    patch_markdown_antes_de_ejecutar_output_dir(nb, cfg["output_dir"])
    prepend_interpretacion_warning(nb)
    append_execution_registry_cells(nb)
    clear_code_cell_outputs(nb)

    out = DST_DIR / (
        f"train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase8_eficiencia_{variant_key}.ipynb"
    )
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for key in ("cpu", "cuda"):
        p = build_notebook(key)
        print("Escrito:", p)


if __name__ == "__main__":
    main()

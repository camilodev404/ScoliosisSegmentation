# -*- coding: utf-8 -*-
"""
Genera notebooks Fase 4 — augmentación geométrica suave en ROI multiclase (solo train).

Parte del cascada V3 local (ya con Fase 3 integrada: LR multiclase + cosine + cudnn reproducible).
Activa `apply_roi_augment=True` en `CascadedThoracolumbarDataset` de **train** únicamente.

Desde la raíz del repo:
  python notebooks/v3_local/mejoras/scripts/build_fase4_augment_roi_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

from cascade_v3_mejora_notebook_common import (
    COLAB_CUDA_PROJECT_ROOT,
    append_execution_registry_cells,
    clear_code_cell_outputs,
    insert_markdown_before_read_gray,
    lines_from_str,
    patch_config_cell_for_training_variant,
    patch_markdown_antes_de_ejecutar_output_dir,
    prepend_interpretacion_warning,
    prepend_mapa_and_optional_colab_cuda_cells,
    replace_cascade_checkpoint_paths,
)

SRC = Path(__file__).resolve().parents[2] / "train_spine_cascade_binary_to_thoracolumbar_v3.ipynb"
DST_DIR = Path(__file__).resolve().parents[1] / "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase4_augmentacion_roi"

TRAIN_DS_OFF = """multiclass_train_ds = CascadedThoracolumbarDataset(
    multiclass_splits_df.query("partition == 'train'"),
    image_size=IMG_SIZE_MULTICLASS,
    roi_mode='gt_binary',
    roi_lookup=None,
    apply_jitter=True,
    apply_roi_augment=False,
)"""

TRAIN_DS_ON = """multiclass_train_ds = CascadedThoracolumbarDataset(
    multiclass_splits_df.query("partition == 'train'"),
    image_size=IMG_SIZE_MULTICLASS,
    roi_mode='gt_binary',
    roi_lookup=None,
    apply_jitter=True,
    apply_roi_augment=True,
)"""

VARIANTS: dict[str, dict] = {
    "cpu": {
        "label": "CPU (perfil liviano)",
        "output_dir": "training_runs_cascade_v3_fase4_augment_roi_cpu",
        "binary_pt": "binary_spine_cascade_fase4_augment_roi_cpu_best.pt",
        "multiclass_pt": "thoracolumbar_{MULTICLASS_SUBSET}_cascade_fase4_augment_roi_cpu_best.pt",
        "img_binary": "(320, 160)",
        "img_mc": "(320, 160)",
        "batch": "2",
        "binary_epochs": "6",
        "multiclass_epochs": "8",
        "nota": (
            "Menor resolución y menos épocas. Augment ROI en train (rotación ±4°, escala 0,98–1,02, 50% sin cambio). "
            "Métricas no comparables 1:1 con perfil CUDA 512×256."
        ),
    },
    "cuda": {
        "label": "CUDA (perfil completo)",
        "output_dir": "training_runs_cascade_v3_fase4_augment_roi_cuda",
        "binary_pt": "binary_spine_cascade_fase4_augment_roi_cuda_best.pt",
        "multiclass_pt": "thoracolumbar_{MULTICLASS_SUBSET}_cascade_fase4_augment_roi_cuda_best.pt",
        "img_binary": "(512, 256)",
        "img_mc": "(512, 256)",
        "batch": "4",
        "binary_epochs": "14",
        "multiclass_epochs": "24",
        "nota": (
            "Misma escala y épocas que el cascada V3 actual (incl. Fase 3 en multiclase). "
            "Único cambio experimental: `apply_roi_augment=True` en train multiclase (`apply_fase4_roi_geom_augment_uint8`)."
        ),
    },
}


def patch_train_roi_augment_on(nb: dict) -> None:
    patched = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s = "".join(cell["source"])
        if TRAIN_DS_ON in s and "multiclass_train_ds = CascadedThoracolumbarDataset" in s:
            patched = True
            break
        if TRAIN_DS_OFF in s:
            cell["source"] = lines_from_str(s.replace(TRAIN_DS_OFF, TRAIN_DS_ON, 1))
            patched = True
            break
    if not patched:
        raise RuntimeError("No se encontro multiclass_train_ds para augment ROI (OFF u ON)")


def build_notebook(variant_key: str) -> Path:
    cfg = VARIANTS[variant_key]
    nb = json.loads(SRC.read_text(encoding="utf-8"))

    mapa_md = f"""# Fase 4 — Augmentación geométrica suave en ROI multiclase (`_{variant_key}`)

## Finalidad de la mejora

Según el plan (`PLAN_ACCION_AJUSTES_MODELOS.md` §3 Fase 4), rotación y escala **pequeñas** en el recorte ROI multiclase (solo **train**) pueden mejorar generalización sin distorsionar de forma agresiva las radiografías.

## Mapa de cambios (vs `train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` vivo)

| Ubicación | Qué cambia |
|-----------|-------------|
| Esta introducción | Contexto y mapa (no existe en el base). |
| Celda markdown **antes** de `def read_gray` | Marca **[FASE 4]** (referencia). |
| Función `apply_fase4_roi_geom_augment_uint8` + `prepare_multiclass_cascade_sample` | Ya están en el notebook **vivo** con `apply_roi_augment` por defecto **False**; aquí se activa **solo** en `multiclass_train_ds`. |
| Config (`OUTPUT_DIR`, épocas, `IMG_*`, `BATCH_SIZE`, rutas `.pt`) | **Perfil {cfg["label"]}** — ver celda de configuración. |
| Binario, scheduler Fase 3, letterbox | Igual que el cascada V3 vivo (sin Fase 1 ni Fase 2). |

## Variante de ejecución: **{cfg["label"]}**

{cfg["nota"]}

**Augment (train multiclase):** rotación uniforme en **±4°**, escala **0,98–1,02**, **50%** de probabilidad de no aplicar transformación en cada muestra.

---

"""
    if variant_key == "cuda":
        mapa_md += (
            "## Colab — raíz por defecto (`_cuda`)\n\n"
            "Si el repo llega a Drive como **Other computers / Mi portátil / ScoliosisSegmentation**, "
            "la celda de configuración prueba **antes** que `MyDrive` la ruta:\n\n"
            f"`{COLAB_CUDA_PROJECT_ROOT.as_posix()}`\n\n"
            "Si tu carpeta tiene otro nombre, usa `MAIA_PROJECT_ROOT` o `%cd` a la raíz del clon.\n\n"
            "---\n\n"
        )
    prepend_mapa_and_optional_colab_cuda_cells(nb, variant_key, mapa_md)

    insert_markdown_before_read_gray(
        nb,
        lines_from_str(
            "### [FASE 4 — AUGMENT ROI] Referencia\n\n"
            "- La augmentación se aplica **solo** en la partición **train** del multiclase (`apply_roi_augment=True`).\n"
            "- Val/test **no** usan augment (`apply_roi_augment=False`).\n"
            "- Implementación: `apply_fase4_roi_geom_augment_uint8` en la misma celda que `prepare_multiclass_cascade_sample`.\n"
        ),
    )

    patch_config_cell_for_training_variant(nb, cfg, variant_key, "FASE 4 augment ROI")
    patch_train_roi_augment_on(nb)
    replace_cascade_checkpoint_paths(nb, cfg)
    patch_markdown_antes_de_ejecutar_output_dir(nb, cfg["output_dir"])
    prepend_interpretacion_warning(nb)
    append_execution_registry_cells(nb)
    clear_code_cell_outputs(nb)

    out = DST_DIR / f"train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase4_augmentacion_roi_{variant_key}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for key in ("cpu", "cuda"):
        p = build_notebook(key)
        print("Escrito:", p)


if __name__ == "__main__":
    main()

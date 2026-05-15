# -*- coding: utf-8 -*-
"""
Genera dos notebooks Fase 2 (refuerzo CE T7–T12) desde cascada V3 base.

**Alcance:** solo multiclase (pesos CE T7–T12). Plantilla común en
`cascade_v3_mejora_notebook_common.py`.

Ejecutar desde la raíz del repo:
  python notebooks/v3_local/mejoras/scripts/build_fase2_pesos_t7_t12_notebooks.py
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
DST_DIR = (
    Path(__file__).resolve().parents[1]
    / "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase2_pesos_clases_t7_t12"
)

OLD_CLASS_WEIGHTS_BLOCK = """class_weights, class_weights_df = estimate_multiclass_class_weights(
    multiclass_splits_df.query("partition == 'train'")
)
multiclass_loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)"""

VARIANTS: dict[str, dict] = {
    "cpu": {
        "label": "CPU (perfil liviano)",
        "output_dir": "training_runs_cascade_v3_fase2_pesos_t7_t12_cpu",
        "binary_pt": "binary_spine_cascade_fase2_pesos_t7_t12_cpu_best.pt",
        "multiclass_pt": "thoracolumbar_{MULTICLASS_SUBSET}_cascade_fase2_pesos_t7_t12_cpu_best.pt",
        "img_binary": "(320, 160)",
        "img_mc": "(320, 160)",
        "batch": "2",
        "binary_epochs": "6",
        "multiclass_epochs": "8",
        "f2_t7_t12_mult": "1.25",
        "nota": (
            "Menor resolucion y menos epocas para iterar sin CUDA. "
            "Las metricas **no son comparables 1:1** con el baseline 512x256; sirven para tendencia y depuracion."
        ),
    },
    "cuda": {
        "label": "CUDA (perfil completo)",
        "output_dir": "training_runs_cascade_v3_fase2_pesos_t7_t12_cuda",
        "binary_pt": "binary_spine_cascade_fase2_pesos_t7_t12_cuda_best.pt",
        "multiclass_pt": "thoracolumbar_{MULTICLASS_SUBSET}_cascade_fase2_pesos_t7_t12_cuda_best.pt",
        "img_binary": "(512, 256)",
        "img_mc": "(512, 256)",
        "batch": "4",
        "binary_epochs": "14",
        "multiclass_epochs": "24",
        "f2_t7_t12_mult": "1.40",
        "nota": (
            "Misma escala y epocas que el cascada V3 base; requiere GPU con PyTorch compatible. "
            "Metricas comparables al baseline en condiciones similares."
        ),
    },
}


def _new_class_weights_block(cfg: dict) -> str:
    m = cfg["f2_t7_t12_mult"]
    return f"""class_weights, class_weights_df = estimate_multiclass_class_weights(
    multiclass_splits_df.query("partition == 'train'")
)
# --- [FASE 2 / T7-T12] Multiplicador extra sobre pesos CE (class_id 7..12 = T7..T12; ver class_names) ---
_F2_T7_T12_MULT = {m}
for _cid in range(7, 13):
    class_weights[_cid] = class_weights[_cid] * _F2_T7_T12_MULT
    _m = class_weights_df["class_id"] == _cid
    class_weights_df.loc[_m, "weight"] = class_weights_df.loc[_m, "weight"] * _F2_T7_T12_MULT
multiclass_loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)"""


def build_notebook(variant_key: str) -> Path:
    cfg = VARIANTS[variant_key]
    nb = json.loads(SRC.read_text(encoding="utf-8"))

    mapa_md = f"""# Fase 2 — Refuerzo de pesos CE en T7–T12 (`_{variant_key}`)

## Finalidad de la mejora

Tras `estimate_multiclass_class_weights`, se aplica un **multiplicador adicional** a los `class_id` **7–12** (vertebras **T7–T12**) en el tensor pasado a `nn.CrossEntropyLoss` y en `class_weights_df` (CSV / display), sin tocar el resto del pipeline ni el letterbox de Fase 1.

## Mapa de cambios (vs `train_spine_cascade_binary_to_thoracolumbar_v3.ipynb`)

| Ubicacion | Que cambia |
|-----------|------------|
| Esta introduccion | Contexto y mapa (no existe en el base). |
| Celda markdown **inmediatamente antes** de `def read_gray` | Marca **[FASE 2]** (referencia; el codigo tocado esta mas abajo). |
| Tras `estimate_multiclass_class_weights(...)` | **Boost T7–T12** antes de instanciar `CrossEntropyLoss`. |
| Config (`OUTPUT_DIR`, epocas, `IMG_*`, `BATCH_SIZE`, rutas `.pt`) | **Perfil {cfg["label"]}** — ver celda de configuracion siguiente. |
| Resto (split, U-Net, letterbox ausente, metricas CSV) | Igual que el cascada V3 base. |

## Variante de ejecucion: **{cfg["label"]}**

{cfg["nota"]}

**Multiplicador T7–T12 en esta variante:** `{cfg["f2_t7_t12_mult"]}` (ajustable en el script de build si hace falta otro valor).

---

"""
    if variant_key == "cuda":
        mapa_md += (
            "## Colab — raiz por defecto (`_cuda`)\n\n"
            "Si el repo llega a Drive como **Other computers / Mi portátil / ScoliosisSegmentation**, "
            "la celda de configuracion prueba **antes** que `MyDrive` la ruta:\n\n"
            f"`{COLAB_CUDA_PROJECT_ROOT.as_posix()}`\n\n"
            "Si tu carpeta tiene otro nombre, usa `MAIA_PROJECT_ROOT` o `%cd` a la raiz del clon.\n\n"
            "---\n\n"
        )
    prepend_mapa_and_optional_colab_cuda_cells(nb, variant_key, mapa_md)

    insert_markdown_before_read_gray(
        nb,
        lines_from_str(
            "### [FASE 2 — PESOS T7–T12] Referencia\n\n"
            "- El cambio de codigo esta en la celda de **entrenamiento multiclase**: despues de "
            "`estimate_multiclass_class_weights` y **antes** de `nn.CrossEntropyLoss`.\n"
            "- Se multiplican los pesos CE de `class_id` 7–12 (T7–T12) por `_F2_T7_T12_MULT`.\n"
            "- No se modifica `dice_loss_multiclass` en esta fase (solo refuerzo en la rama CE).\n"
        ),
    )

    patch_config_cell_for_training_variant(nb, cfg, variant_key, "FASE 2 pesos T7-T12")

    patched_weights = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s = "".join(cell["source"])
        if OLD_CLASS_WEIGHTS_BLOCK in s:
            s2 = s.replace(OLD_CLASS_WEIGHTS_BLOCK, _new_class_weights_block(cfg), 1)
            cell["source"] = lines_from_str(s2)
            patched_weights = True
            break
    if not patched_weights:
        raise RuntimeError("No se encontro el bloque class_weights / CrossEntropyLoss para Fase 2")

    replace_cascade_checkpoint_paths(nb, cfg)
    patch_markdown_antes_de_ejecutar_output_dir(nb, cfg["output_dir"])
    prepend_interpretacion_warning(nb)
    append_execution_registry_cells(nb)
    clear_code_cell_outputs(nb)

    out = DST_DIR / f"train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase2_pesos_clases_t7_t12_{variant_key}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for key in ("cpu", "cuda"):
        p = build_notebook(key)
        print("Escrito:", p)


if __name__ == "__main__":
    main()

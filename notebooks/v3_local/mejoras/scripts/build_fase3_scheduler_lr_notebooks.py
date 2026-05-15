# -*- coding: utf-8 -*-
"""
Genera dos notebooks Fase 3 (LR multiclase + cosine annealing) desde cascada V3 base.

**Alcance:** solo multiclase (optimizador + scheduler). Plantilla común en
`cascade_v3_mejora_notebook_common.py`.

Ejecutar desde la raíz del repo:
  python notebooks/v3_local/mejoras/scripts/build_fase3_scheduler_lr_notebooks.py
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
DST_DIR = Path(__file__).resolve().parents[1] / "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase3_scheduler_lr"

OLD_MC_OPTIM_BLOCK = """multiclass_optimizer = torch.optim.AdamW(multiclass_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

multiclass_history = []"""

OLD_MC_PRINT_TAIL = """        f"val_macro_iou={row['val_macro_iou_fg']:.4f}"
    )

multiclass_history_df = pd.DataFrame(multiclass_history)"""

VARIANTS: dict[str, dict] = {
    "cpu": {
        "label": "CPU (perfil liviano)",
        "output_dir": "training_runs_cascade_v3_fase3_scheduler_lr_cpu",
        "binary_pt": "binary_spine_cascade_fase3_scheduler_lr_cpu_best.pt",
        "multiclass_pt": "thoracolumbar_{MULTICLASS_SUBSET}_cascade_fase3_scheduler_lr_cpu_best.pt",
        "img_binary": "(320, 160)",
        "img_mc": "(320, 160)",
        "batch": "2",
        "binary_epochs": "6",
        "multiclass_epochs": "8",
        "mc_lr_scale": "0.75",
        "nota": (
            "Menor resolucion y menos epocas. `LR_MULTICLASS = LR * 0,75` y cosine annealing en multiclase. "
            "Metricas no comparables 1:1 con baseline 512x256."
        ),
    },
    "cuda": {
        "label": "CUDA (perfil completo)",
        "output_dir": "training_runs_cascade_v3_fase3_scheduler_lr_cuda",
        "binary_pt": "binary_spine_cascade_fase3_scheduler_lr_cuda_best.pt",
        "multiclass_pt": "thoracolumbar_{MULTICLASS_SUBSET}_cascade_fase3_scheduler_lr_cuda_best.pt",
        "img_binary": "(512, 256)",
        "img_mc": "(512, 256)",
        "batch": "4",
        "binary_epochs": "14",
        "multiclass_epochs": "24",
        "mc_lr_scale": "0.5",
        "nota": (
            "Misma escala y epocas que el cascada V3 base. "
            "`LR_MULTICLASS = LR * 0,5` y `CosineAnnealingLR(T_max=MULTICLASS_EPOCHS, eta_min≈1% del LR inicial)`; "
            "checkpoint del multiclase sigue eligiendose por mejor `val_macro_dice_fg`."
        ),
    },
}


def _new_mc_optim_block(cfg: dict) -> str:
    sc = cfg["mc_lr_scale"]
    return f"""# --- [FASE 3] LR multiclase escalado + CosineAnnealingLR (solo etapa multiclase) ---
_MULTICLASS_LR_SCALE = {sc}
LR_MULTICLASS = LR * _MULTICLASS_LR_SCALE
multiclass_optimizer = torch.optim.AdamW(
    multiclass_model.parameters(), lr=LR_MULTICLASS, weight_decay=WEIGHT_DECAY
)
multiclass_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    multiclass_optimizer, T_max=MULTICLASS_EPOCHS, eta_min=max(float(1e-8), LR_MULTICLASS * 0.01)
)

multiclass_history = []"""


NEW_MC_PRINT_TAIL = """        f"val_macro_iou={row['val_macro_iou_fg']:.4f}"
    )
    multiclass_scheduler.step()

multiclass_history_df = pd.DataFrame(multiclass_history)"""


def build_notebook(variant_key: str) -> Path:
    cfg = VARIANTS[variant_key]
    nb = json.loads(SRC.read_text(encoding="utf-8"))

    mapa_md = f"""# Fase 3 — Optimización multiclase: LR + scheduler (`_{variant_key}`)

## Finalidad de la mejora

Según el plan (`PLAN_ACCION_AJUSTES_MODELOS.md` §3 Fase 3), reducir el **learning rate** de la etapa **multiclase** y aplicar **cosine annealing** puede estabilizar el entrenamiento sobre ROI y limitar sobreajuste a bordes, **sin** cambiar el flujo binario → ROI → multiclase ni el criterio de mejor modelo (**mejor `val_macro_dice_fg`**).

## Mapa de cambios (vs `train_spine_cascade_binary_to_thoracolumbar_v3.ipynb`)

| Ubicacion | Que cambia |
|-----------|------------|
| Esta introduccion | Contexto y mapa (no existe en el base). |
| Celda markdown **antes** de `def read_gray` | Marca **[FASE 3]** (referencia). |
| Tras crear `multiclass_optimizer` | **LR_MULTICLASS**, **CosineAnnealingLR** y `scheduler.step()` por epoca (solo multiclase). |
| Config (`OUTPUT_DIR`, epocas, `IMG_*`, `BATCH_SIZE`, rutas `.pt`) | **Perfil {cfg["label"]}** — ver celda de configuracion. |
| Binario, pesos CE por clase, letterbox | Igual que el cascada V3 base (sin Fase 1 ni Fase 2). |

## Variante de ejecucion: **{cfg["label"]}**

{cfg["nota"]}

**Escala LR multiclase (`LR_MULTICLASS = LR * …`):** `{cfg["mc_lr_scale"]}` (editable en `VARIANTS` del script de build).

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
            "### [FASE 3 — SCHEDULER LR] Referencia\n\n"
            "- El cambio esta en la celda de **entrenamiento multiclase**: optimizador con `lr=LR_MULTICLASS` "
            "y `CosineAnnealingLR` con `T_max=MULTICLASS_EPOCHS`.\n"
            "- Al final de cada epoca de multiclase se llama `multiclass_scheduler.step()`.\n"
            "- La etapa **binaria** sigue usando `LR` sin scheduler en esta fase.\n"
        ),
    )

    patch_config_cell_for_training_variant(nb, cfg, variant_key, "FASE 3 scheduler LR")

    patched = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s = "".join(cell["source"])
        if OLD_MC_OPTIM_BLOCK not in s:
            continue
        s2 = s.replace(OLD_MC_OPTIM_BLOCK, _new_mc_optim_block(cfg), 1)
        if OLD_MC_PRINT_TAIL not in s2:
            raise RuntimeError("patron cola print multiclase no encontrado tras parche optimizer")
        s2 = s2.replace(OLD_MC_PRINT_TAIL, NEW_MC_PRINT_TAIL, 1)
        cell["source"] = lines_from_str(s2)
        patched = True
        break
    if not patched:
        raise RuntimeError("No se encontro el bloque multiclass_optimizer / multiclass_history para Fase 3")

    replace_cascade_checkpoint_paths(nb, cfg)
    patch_markdown_antes_de_ejecutar_output_dir(nb, cfg["output_dir"])
    prepend_interpretacion_warning(nb)
    append_execution_registry_cells(nb)
    clear_code_cell_outputs(nb)

    out = DST_DIR / f"train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase3_scheduler_lr_{variant_key}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for key in ("cpu", "cuda"):
        p = build_notebook(key)
        print("Escrito:", p)


if __name__ == "__main__":
    main()

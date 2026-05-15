# -*- coding: utf-8 -*-
"""
Genera notebooks Fase 5 — robustez estadística (multiseed) desde el cascada V3 vivo.

Mismo pipeline que `train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` (Fase 3 + 4 integradas),
con **SEED** y carpetas de salida distintas por variante.

Desde la raíz del repo:
  python notebooks/v3_local/mejoras/scripts/build_fase5_multiseed_notebooks.py
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
DST_DIR = Path(__file__).resolve().parents[1] / "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase5_multiseed"

# Mismo perfil CUDA que el base; solo cambian semilla y artefactos.
_CUDA_PROFILE = {
    "img_binary": "(512, 256)",
    "img_mc": "(512, 256)",
    "batch": "4",
    "binary_epochs": "14",
    "multiclass_epochs": "24",
}

SEED_RUNS: list[dict[str, str | int]] = [
    {
        "file_suffix": "cuda_seed42",
        "seed": 42,
        "tag": "s42",
        "output_dir": "training_runs_cascade_v3_fase5_multiseed_cuda_s42",
        "label": "CUDA — semilla 42 (referencia habitual)",
    },
    {
        "file_suffix": "cuda_seed1337",
        "seed": 1337,
        "tag": "s1337",
        "output_dir": "training_runs_cascade_v3_fase5_multiseed_cuda_s1337",
        "label": "CUDA — semilla 1337",
    },
    {
        "file_suffix": "cuda_seed4242",
        "seed": 4242,
        "tag": "s4242",
        "output_dir": "training_runs_cascade_v3_fase5_multiseed_cuda_s4242",
        "label": "CUDA — semilla 4242",
    },
]

SEED_BLOCK_OLD = "SEED = 42\nrandom.seed(SEED)\nnp.random.seed(SEED)"


def _cfg_from_run(run: dict[str, str | int]) -> dict[str, str]:
    tag = str(run["tag"])
    return {
        "label": str(run["label"]),
        "output_dir": str(run["output_dir"]),
        "binary_pt": f"binary_spine_cascade_fase5_multiseed_{tag}_best.pt",
        "multiclass_pt": f"thoracolumbar_{{MULTICLASS_SUBSET}}_cascade_fase5_multiseed_{tag}_best.pt",
        **_CUDA_PROFILE,
        "nota": (
            "Misma lógica que el notebook vivo (Fase 3 + Fase 4). "
            f"Único cambio experimental: `SEED = {run['seed']}` y rutas de salida propias."
        ),
    }


def patch_seed_block(nb: dict, seed: int) -> None:
    new_block = f"SEED = {seed}\nrandom.seed(SEED)\nnp.random.seed(SEED)"
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s = "".join(cell["source"])
        if SEED_BLOCK_OLD not in s:
            continue
        cell["source"] = lines_from_str(s.replace(SEED_BLOCK_OLD, new_block, 1))
        return
    raise RuntimeError("No se encontro el bloque SEED/random/np.random esperado")


def build_notebook(run: dict[str, str | int]) -> Path:
    cfg = _cfg_from_run(run)
    seed = int(run["seed"])
    nb = json.loads(SRC.read_text(encoding="utf-8"))

    mapa_md = f"""# Fase 5 — Robustez estadística / multiseed (`{run['file_suffix']}`)

## Finalidad

Según el plan (`PLAN_ACCION_AJUSTES_MODELOS.md` §3 Fase 5), repetir el **mismo** procedimiento cascada ya integrado (Fase 3 + Fase 4 en el notebook vivo) con **otra semilla** para comprobar si las métricas de test se **replican** en dirección y magnitud.

## Qué cambia en este notebook

| Aspecto | Valor |
|---------|--------|
| `SEED` | **{seed}** |
| `OUTPUT_DIR` (relativo a `analysis_outputs_v3`) | `{cfg['output_dir']}` |
| Checkpoints en `MODEL_DIR` | sufijo **`{run['tag']}`** en los `.pt` |

El resto (épocas, resolución, augment ROI train, LR multiclase + cosine, `cudnn` determinista) coincide con `train_spine_cascade_binary_to_thoracolumbar_v3.ipynb`.

## Variante: **{cfg['label']}**

{cfg['nota']}

---

"""
    mapa_md += (
        "## Colab — raíz por defecto (`_cuda`)\n\n"
        "Si el repo llega a Drive como **Other computers / Mi portátil / ScoliosisSegmentation**, "
        "la celda de configuración prueba **antes** que `MyDrive` la ruta:\n\n"
        f"`{COLAB_CUDA_PROJECT_ROOT.as_posix()}`\n\n"
        "Si tu carpeta tiene otro nombre, usa `MAIA_PROJECT_ROOT` o `%cd` a la raíz del clon.\n\n"
        "---\n\n"
    )
    prepend_mapa_and_optional_colab_cuda_cells(nb, "cuda", mapa_md)

    insert_markdown_before_read_gray(
        nb,
        lines_from_str(
            f"### [FASE 5 — MULTISEED] Semilla **{seed}**\n\n"
            "- Mismo código que el cascada V3 vivo salvo `SEED` y rutas de esta variante.\n"
            "- Tras ejecutar las tres variantes, comparar `macro_dice_fg` / `macro_iou_fg` en test y decidir según el plan (replicación vs «prometedor, no consolidado»).\n"
        ),
    )

    patch_seed_block(nb, seed)
    patch_config_cell_for_training_variant(nb, cfg, "cuda", f"FASE 5 multiseed {run['tag']}")
    replace_cascade_checkpoint_paths(nb, cfg)
    patch_markdown_antes_de_ejecutar_output_dir(nb, cfg["output_dir"])
    prepend_interpretacion_warning(nb)
    append_execution_registry_cells(nb)
    clear_code_cell_outputs(nb)

    fname = f"train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase5_multiseed_{run['file_suffix']}.ipynb"
    out = DST_DIR / fname
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for run in SEED_RUNS:
        p = build_notebook(run)
        print("Escrito:", p)


if __name__ == "__main__":
    main()

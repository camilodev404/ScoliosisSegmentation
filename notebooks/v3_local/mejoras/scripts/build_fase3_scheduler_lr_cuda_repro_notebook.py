# -*- coding: utf-8 -*-
"""
Genera el notebook de **comprobación / reproducibilidad** del `_cuda` de Fase 3.

Mismo código de entrenamiento que `train_spine_cascade_…_mejorafase3_scheduler_lr_cuda.ipynb`,
pero con **OUTPUT_DIR** y **checkpoints** distintos para no pisar el run ya ejecutado (run A).

Ejecutar desde la raíz del repo:
  python notebooks/v3_local/mejoras/scripts/build_fase3_scheduler_lr_cuda_repro_notebook.py
"""
from __future__ import annotations

import importlib.util
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

_scripts = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "build_fase3_scheduler_lr_notebooks",
    _scripts / "build_fase3_scheduler_lr_notebooks.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

SRC: Path = _mod.SRC
DST_DIR: Path = _mod.DST_DIR
OLD_MC_OPTIM_BLOCK: str = _mod.OLD_MC_OPTIM_BLOCK
VARIANTS: dict = _mod.VARIANTS
_new_mc_optim_block = _mod._new_mc_optim_block

RUN_A_DIR = "training_runs_cascade_v3_fase3_scheduler_lr_cuda"
RUN_B_DIR = "training_runs_cascade_v3_fase3_scheduler_lr_cuda_repro_b"

CFG_REPRO: dict[str, str] = {
    **VARIANTS["cuda"],
    "output_dir": RUN_B_DIR,
    "binary_pt": "binary_spine_cascade_fase3_scheduler_lr_cuda_repro_b_best.pt",
    "multiclass_pt": "thoracolumbar_{MULTICLASS_SUBSET}_cascade_fase3_scheduler_lr_cuda_repro_b_best.pt",
    "nota": (
        "**Verificación (run B):** idéntico al `_cuda` de Fase 3 en arquitectura, épocas, batch, "
        "`LR_MULTICLASS = LR × 0,5` y `CosineAnnealingLR`. Solo cambian rutas de artefactos respecto al run A."
    ),
}


def _append_cudnn_determinism(nb: dict) -> None:
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s = "".join(cell["source"])
        if "torch.cuda.manual_seed_all(SEED)" not in s:
            continue
        if "cudnn.benchmark" in s:
            return
        insert = (
            "\n\n"
            "# --- [FASE 3 — verificación repro] Más determinismo en GPU (puede reducir velocidad) ---\n"
            "if torch.cuda.is_available():\n"
            "    torch.backends.cudnn.benchmark = False\n"
            "    torch.backends.cudnn.deterministic = True\n"
        )
        cell["source"] = lines_from_str(s.replace("torch.cuda.manual_seed_all(SEED)", "torch.cuda.manual_seed_all(SEED)" + insert, 1))
        return
    raise RuntimeError("No se encontro torch.cuda.manual_seed_all(SEED) para inyectar flags cudnn")


def build_repro_notebook() -> Path:
    nb = json.loads(SRC.read_text(encoding="utf-8"))

    mapa_md = f"""# Fase 3 — Verificación de reproducibilidad (`_cuda` run B)

## Objetivo

Comprobar si la decisión **«Adoptar con condición»** del experimento Fase 3 `_cuda` se mantiene cuando se **repite el mismo entrenamiento** con otra carpeta de salida. En el cierre anterior, el **binario en test** bajó levemente respecto al baseline sin cambiar el LR del binario; hace falta ver si eso **se repite** (señal sistemática) o **no** (ruido / no-determinismo).

## De qué es copia este notebook

- **Plantilla de código:** la misma que `train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase3_scheduler_lr_cuda.ipynb` (LR multiclase ×**0,5** + **CosineAnnealingLR** + `scheduler.step()` por época en multiclase; binario sin scheduler).
- **Semilla:** sigue siendo `SEED = 42` como en el base (no se altera a propósito para que la comparación A↔B sea justa).

## Qué cambia respecto al `_cuda` ya ejecutado (run A)

| Aspecto | Run A (ya ejecutado) | Run B (este notebook) |
|--------|----------------------|------------------------|
| `OUTPUT_DIR` (relativo a `analysis_outputs_v3`) | `{RUN_A_DIR}` | `{RUN_B_DIR}` |
| Checkpoints `.pt` en `MODEL_DIR` | `…_fase3_scheduler_lr_cuda_best.pt` | `…_fase3_scheduler_lr_cuda_repro_b_best.pt` |
| Inyección extra | — | Tras fijar semilla CUDA: `cudnn.benchmark = False`, `cudnn.deterministic = True` para acercar resultados entre ejecuciones (no garantiza determinismo total en todos los kernels). |

## Contra qué comparar al terminar (orden recomendado)

1. **Run A vs run B** (`{RUN_A_DIR}` vs `{RUN_B_DIR}`): si binario y macro quedan muy parecidos, la variación observada en el primer run es probablemente **ruido**. Si divergen mucho, conviene más repetición o revisar entorno.
2. **Run B vs baseline cascada** (`training_runs_cascade_v3/`): criterio de adopción global del plan (¿sigue mejorando el multiclase macro frente al base?).
3. **Run A vs baseline** (ya documentado en `RESULTADOS_Y_DECISIONES_GENERAL.md`): contexto histórico; no sustituye el punto 1.

## Colab — raiz por defecto (`_cuda`)

Si el repo llega a Drive como **Other computers / Mi portátil / ScoliosisSegmentation**, la celda de configuración prueba **antes** que `MyDrive` la ruta:

`{COLAB_CUDA_PROJECT_ROOT.as_posix()}`

Si tu carpeta tiene otro nombre, usa `MAIA_PROJECT_ROOT` o `%cd` a la raiz del clon.

---

"""
    prepend_mapa_and_optional_colab_cuda_cells(nb, "cuda", mapa_md)

    insert_markdown_before_read_gray(
        nb,
        lines_from_str(
            "### [FASE 3 — RUN B / REPRO] Referencia rápida\n\n"
            "- Mismo parche multiclase que el notebook `…_mejorafase3_scheduler_lr_cuda.ipynb`.\n"
            "- Los CSV de test/val deben generarse bajo `…/" + RUN_B_DIR + "/`.\n"
            "- Tras ejecutar: registrar en `experiment_registry.csv` (p. ej. id `3_cuda_repro_b`) y actualizar la ficha Fase 3 en `RESULTADOS_Y_DECISIONES_GENERAL.md`.\n"
        ),
    )

    patch_config_cell_for_training_variant(nb, CFG_REPRO, "cuda", "FASE 3 repro verificacion")

    patched = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s = "".join(cell["source"])
        if OLD_MC_OPTIM_BLOCK not in s:
            continue
        s2 = s.replace(OLD_MC_OPTIM_BLOCK, _new_mc_optim_block(CFG_REPRO), 1)
        if _mod.OLD_MC_PRINT_TAIL not in s2:
            raise RuntimeError("patron cola print multiclase no encontrado tras parche optimizer")
        s2 = s2.replace(_mod.OLD_MC_PRINT_TAIL, _mod.NEW_MC_PRINT_TAIL, 1)
        cell["source"] = lines_from_str(s2)
        patched = True
        break
    if not patched:
        raise RuntimeError("No se encontro el bloque multiclass_optimizer para Fase 3 repro")

    replace_cascade_checkpoint_paths(nb, CFG_REPRO)
    patch_markdown_antes_de_ejecutar_output_dir(nb, CFG_REPRO["output_dir"])
    prepend_interpretacion_warning(nb)
    _append_cudnn_determinism(nb)
    append_execution_registry_cells(nb)
    clear_code_cell_outputs(nb)

    out = DST_DIR / (
        "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase3_scheduler_lr_cuda_repro_verificacion.ipynb"
    )
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    p = build_repro_notebook()
    print("Escrito:", p)


if __name__ == "__main__":
    main()

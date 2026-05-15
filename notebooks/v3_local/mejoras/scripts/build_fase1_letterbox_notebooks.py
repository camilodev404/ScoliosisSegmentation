# -*- coding: utf-8 -*-
"""
Genera dos notebooks Fase 1 (letterbox) desde cascada V3:

  * ..._mejorafase1_letterbox_roi_cpu.ipynb   — perfil liviano (CPU / pocos recursos)
  * ..._mejorafase1_letterbox_roi_cuda.ipynb — perfil completo (GPU)

Ejecutar desde la raíz del repo:  python notebooks/v3_local/mejoras/scripts/build_fase1_letterbox_notebooks.py
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
    / "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_letterbox_roi"
)

LETTERBOX_FN = r'''def letterbox_gray_and_mask(
    image_crop_u8: np.ndarray,
    mask_crop_u8: np.ndarray,
    output_size: tuple[int, int],
    mask_pad_value: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Escala manteniendo proporcion y centra en output_size (H, W). Imagen float32 [0,1]; mascara uint8."""
    out_h, out_w = output_size
    in_h, in_w = image_crop_u8.shape[:2]
    if in_h <= 0 or in_w <= 0:
        img = np.zeros((out_h, out_w), dtype=np.float32)
        msk = np.full((out_h, out_w), mask_pad_value, dtype=np.uint8)
        return img, msk
    scale = min(out_h / in_h, out_w / in_w)
    nh = max(1, int(round(in_h * scale)))
    nw = max(1, int(round(in_w * scale)))
    img_small = np.array(Image.fromarray(image_crop_u8).resize((nw, nh), resample=Image.BILINEAR))
    msk_small = np.array(
        Image.fromarray(mask_crop_u8.astype(np.uint8)).resize((nw, nh), resample=Image.NEAREST)
    )
    pad_y0 = (out_h - nh) // 2
    pad_x0 = (out_w - nw) // 2
    out_img = np.zeros((out_h, out_w), dtype=np.uint8)
    out_msk = np.full((out_h, out_w), mask_pad_value, dtype=np.uint8)
    out_img[pad_y0 : pad_y0 + nh, pad_x0 : pad_x0 + nw] = img_small
    out_msk[pad_y0 : pad_y0 + nh, pad_x0 : pad_x0 + nw] = msk_small
    out_img_f = out_img.astype(np.float32) / 255.0
    return out_img_f, out_msk


'''

CROP_TO_SCALEBOX = """def crop_array(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return arr[y0:y1, x0:x1]


def scale_bbox("""

OLD_RESIZE_BLOCK = """    image_crop = resize_image(image_crop, output_size).astype(np.float32) / 255.0
    multiclass_crop = resize_mask(multiclass_crop, output_size).astype(np.int64)
    image_crop = np.expand_dims(image_crop, axis=0)"""

NEW_RESIZE_BLOCK = """    # --- [FASE 1 / LETTERBOX] unico cambio estructural vs cascada V3 base (resto igual) ---
    image_crop, multiclass_crop = letterbox_gray_and_mask(
        image_crop, multiclass_crop, output_size, mask_pad_value=IGNORE_INDEX
    )
    multiclass_crop = multiclass_crop.astype(np.int64)
    image_crop = np.expand_dims(image_crop, axis=0)"""


def patch_cell_helpers(source: str) -> str:
    if "letterbox_gray_and_mask" in source:
        return source
    if CROP_TO_SCALEBOX not in source:
        raise RuntimeError("patron crop_array/scale_bbox no encontrado")
    source = source.replace(
        CROP_TO_SCALEBOX,
        CROP_TO_SCALEBOX.replace("def scale_bbox(", LETTERBOX_FN + "def scale_bbox("),
        1,
    )
    if OLD_RESIZE_BLOCK not in source:
        raise RuntimeError("bloque resize multiclase no encontrado")
    return source.replace(OLD_RESIZE_BLOCK, NEW_RESIZE_BLOCK, 1)


VARIANTS: dict[str, dict] = {
    "cpu": {
        "label": "CPU (perfil liviano)",
        "output_dir": "training_runs_cascade_v3_fase1_letterbox_cpu",
        "binary_pt": "binary_spine_cascade_fase1_letterbox_cpu_best.pt",
        "multiclass_pt": "thoracolumbar_{MULTICLASS_SUBSET}_cascade_fase1_letterbox_cpu_best.pt",
        "img_binary": "(320, 160)",
        "img_mc": "(320, 160)",
        "batch": "2",
        "binary_epochs": "6",
        "multiclass_epochs": "8",
        "nota": (
            "Menor resolucion y menos epocas para iterar en PC sin CUDA viable. "
            "Las metricas **no son comparables 1:1** con el baseline 512x256 sin salvedad; sirven para tendencia y depuracion."
        ),
    },
    "cuda": {
        "label": "CUDA (perfil completo)",
        "output_dir": "training_runs_cascade_v3_fase1_letterbox_cuda",
        "binary_pt": "binary_spine_cascade_fase1_letterbox_cuda_best.pt",
        "multiclass_pt": "thoracolumbar_{MULTICLASS_SUBSET}_cascade_fase1_letterbox_cuda_best.pt",
        "img_binary": "(512, 256)",
        "img_mc": "(512, 256)",
        "batch": "4",
        "binary_epochs": "14",
        "multiclass_epochs": "24",
        "nota": (
            "Misma escala y epocas que el cascada V3 base; requiere GPU con PyTorch compatible. "
            "Metricas comparables al baseline en condiciones similares."
        ),
    },
}


def build_notebook(variant_key: str) -> Path:
    cfg = VARIANTS[variant_key]
    nb = json.loads(SRC.read_text(encoding="utf-8"))

    mapa_md = f"""# Fase 1 — Letterbox en ROI multiclase (`_{variant_key}`)

## Finalidad de la mejora

En la **etapa multiclase sobre ROI**, el notebook base estira (`resize`) el recorte a `IMG_SIZE_MULTICLASS` **deformando** la relacion de aspecto. En esta fase se reemplaza eso por **letterbox**: escala uniforme, centrado y relleno; la mascara se rellena con `IGNORE_INDEX` para no contar en la perdida.

## Mapa de cambios (vs `train_spine_cascade_binary_to_thoracolumbar_v3.ipynb`)

| Ubicacion | Que cambia |
|-----------|------------|
| Esta introduccion | Contexto y mapa (no existe en el base). |
| Celda markdown **inmediatamente antes** de `def read_gray` | Marca **[FASE 1]** donde empieza el codigo tocado. |
| Funcion `letterbox_gray_and_mask` + cuerpo de `prepare_multiclass_cascade_sample` | **Implementacion letterbox** (sustituye `resize_image`/`resize_mask` del crop multiclase). |
| Config (`OUTPUT_DIR`, epocas, `IMG_*`, `BATCH_SIZE`, rutas `.pt`) | **Perfil {cfg["label"]}** — ver celda de configuracion siguiente. |
| Resto (split, U-Net, bucles entrenamiento, metricas CSV) | Igual que el base; revisar textos de interpretacion **tras** correr y pegar numeros nuevos. |

## Variante de ejecucion: **{cfg["label"]}**

{cfg["nota"]}

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
            "### [FASE 1 — LETTERBOX] Implementacion\n\n"
            "- A continuacion esta el bloque de **helpers** del notebook base **mas** la funcion `letterbox_gray_and_mask`.\n"
            "- Dentro de `prepare_multiclass_cascade_sample`, el **unico** cambio respecto al base es el bloque marcado "
            "`[FASE 1 / LETTERBOX]` que sustituye el resize directo del crop multiclase.\n"
            "- El binario y la ROI predicha **no** usan letterbox en esta fase.\n"
        ),
    )

    patch_config_cell_for_training_variant(nb, cfg, variant_key, "FASE 1 letterbox")

    # helpers cell (tras insercion markdown): letterbox en prepare_multiclass_cascade_sample
    ih = None
    for j, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        t = "".join(c.get("source", []))
        if "def read_gray(path: str)" in t and "def letterbox_gray_and_mask" not in t:
            ih = j
            break
    if ih is None:
        raise RuntimeError("celda helpers post-insercion no encontrada")
    nb["cells"][ih]["source"] = lines_from_str(patch_cell_helpers("".join(nb["cells"][ih]["source"])))

    replace_cascade_checkpoint_paths(nb, cfg)
    patch_markdown_antes_de_ejecutar_output_dir(nb, cfg["output_dir"])
    prepend_interpretacion_warning(nb)
    append_execution_registry_cells(nb)
    clear_code_cell_outputs(nb)

    out = DST_DIR / f"train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase1_letterbox_roi_{variant_key}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for key in ("cpu", "cuda"):
        p = build_notebook(key)
        print("Escrito:", p)


if __name__ == "__main__":
    main()

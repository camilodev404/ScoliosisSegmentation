# -*- coding: utf-8 -*-
"""
Genera dos notebooks Fase 1 (letterbox) desde cascada V3:

  * ..._mejorafase1_letterbox_roi_cpu.ipynb   — perfil liviano (CPU / pocos recursos)
  * ..._mejorafase1_letterbox_roi_cuda.ipynb — perfil completo (GPU)

Ejecutar:  python mejoras/scripts/build_fase1_letterbox_notebooks.py
"""
from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path

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


def lines_from_str(s: str) -> list[str]:
    parts = s.splitlines(keepends=True)
    return parts if parts else [""]


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


def insert_markdown_before_read_gray(nb: dict, lines: list[str]) -> None:
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "def read_gray(path: str)" in src or src.strip().startswith("def read_gray"):
            nb["cells"].insert(
                i,
                {
                    "cell_type": "markdown",
                    "id": uuid.uuid4().hex[:8],
                    "metadata": {},
                    "source": lines,
                },
            )
            return
    raise RuntimeError("No se encontro la celda de helpers (read_gray)")


def prepend_interpretacion_warning(nb: dict) -> None:
    for cell in nb["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        s = "".join(cell.get("source", []))
        if "## Como interpretar este notebook al terminar" in s or "## Como interpretar" in s:
            warn = (
                "> **Textos heredados del notebook base:** las secciones de interpretacion / analisis "
                "final deben **actualizarse despues de cada entrenamiento** con las metricas reales de "
                "esta variante (`_cpu` / `_cuda`). No copies conclusiones del V3 base si los numeros cambian.\n\n"
            )
            cell["source"] = lines_from_str(warn) + cell["source"]
            return


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
    nb["cells"][0]["source"] = lines_from_str(mapa_md) + nb["cells"][0]["source"]

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

    # Indice de celda config se desplaza +1 por insercion
    def idx_config() -> int:
        for j, c in enumerate(nb["cells"]):
            if c.get("cell_type") != "code":
                continue
            t = "".join(c.get("source", []))
            if "OUTPUT_DIR = ROOT" in t and "MANIFEST_PATH" in t:
                return j
        raise RuntimeError("celda config no encontrada")

    ic = idx_config()
    c4 = "".join(nb["cells"][ic]["source"])
    if "for _cand in [ROOT, *ROOT.parents]" not in c4:
        legacy = """ROOT = Path.cwd()
if not (ROOT / 'Scoliosis_Dataset_V3' / 'Scoliosis_Dataset').exists() and (ROOT.parent / 'Scoliosis_Dataset_V3' / 'Scoliosis_Dataset').exists():
    ROOT = ROOT.parent
"""
        new_root = """import os


def _resolve_maia_project_root(marker: Path) -> Path:
    \"\"\"Raiz del repo (carpeta que contiene marker). Local, subcarpetas, Colab+Drive.\"\"\"
    env = os.environ.get('MAIA_PROJECT_ROOT', '').strip()
    if env:
        p = Path(env).expanduser().resolve()
        if (p / marker).exists():
            return p
    cwd = Path.cwd().resolve()
    for cand in [cwd, *cwd.parents]:
        if (cand / marker).exists():
            return cand
    drive_nb = Path('/content/drive/MyDrive/Colab Notebooks')
    if drive_nb.is_dir():
        direct = drive_nb / 'MAIA-PROYECTO'
        if (direct / marker).exists():
            return direct.resolve()
        for child in drive_nb.iterdir():
            try:
                if child.is_dir() and (child / marker).exists():
                    return child.resolve()
            except OSError:
                continue
    drive_maia = Path('/content/drive/MyDrive/MAIA-PROYECTO')
    if (drive_maia / marker).exists():
        return drive_maia.resolve()
    raise FileNotFoundError(
        f"No se encontro {marker.as_posix()}. Opciones: (1) %cd a la raiz del repo; "
        f"(2) os.environ['MAIA_PROJECT_ROOT'] = r'.../MAIA-PROYECTO'. cwd={cwd}"
    )


ROOT = _resolve_maia_project_root(Path('Scoliosis_Dataset_V3') / 'Scoliosis_Dataset')
"""
        if legacy in c4:
            c4 = c4.replace(legacy, new_root, 1)

    c4 = c4.replace(
        "OUTPUT_DIR = ROOT / 'analysis_outputs_v3' / 'training_runs_cascade_v3'",
        f"OUTPUT_DIR = ROOT / 'analysis_outputs_v3' / '{cfg['output_dir']}'",
    )
    c4 = c4.replace("IMG_SIZE_BINARY = (512, 256)", f"IMG_SIZE_BINARY = {cfg['img_binary']}")
    c4 = c4.replace("IMG_SIZE_MULTICLASS = (512, 256)", f"IMG_SIZE_MULTICLASS = {cfg['img_mc']}")
    c4 = c4.replace("BATCH_SIZE = 4", f"BATCH_SIZE = {cfg['batch']}")
    c4 = c4.replace("BINARY_EPOCHS = 14", f"BINARY_EPOCHS = {cfg['binary_epochs']}")
    c4 = c4.replace("MULTICLASS_EPOCHS = 24", f"MULTICLASS_EPOCHS = {cfg['multiclass_epochs']}")
    tag = f"FASE 1 letterbox ({variant_key})"
    if tag not in c4:
        c4 = c4.replace(
            "print('ROOT:', ROOT)",
            f"print('{tag} — OUTPUT_DIR:', OUTPUT_DIR)\nprint('ROOT:', ROOT)",
        )
    nb["cells"][ic]["source"] = lines_from_str(c4)

    # helpers cell (tras insercion markdown): parchear antes de insertar letterbox
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

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s = "".join(cell["source"])
        s2 = s.replace(
            "binary_model_path = MODEL_DIR / 'binary_spine_cascade_best.pt'",
            f"binary_model_path = MODEL_DIR / '{cfg['binary_pt']}'",
        )
        s2 = s2.replace(
            "multiclass_model_path = MODEL_DIR / f'thoracolumbar_{MULTICLASS_SUBSET}_cascade_best.pt'",
            f"multiclass_model_path = MODEL_DIR / f'{cfg['multiclass_pt']}'",
        )
        if s2 != s:
            cell["source"] = lines_from_str(s2)

    # Markdown "Antes de ejecutar": ruta salidas
    for cell in nb["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        s = "".join(cell.get("source", []))
        if "## Antes de ejecutar" in s:
            s2 = s.replace(
                "`analysis_outputs_v3/training_runs_cascade_v3/`",
                f"`analysis_outputs_v3/{cfg['output_dir']}/`",
            )
            cell["source"] = lines_from_str(s2)
            break

    prepend_interpretacion_warning(nb)

    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

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

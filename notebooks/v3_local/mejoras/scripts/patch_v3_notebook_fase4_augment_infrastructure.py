# -*- coding: utf-8 -*-
"""
Añade en el notebook cascada V3 la infraestructura Fase 4 (augment ROI multiclase),
desactivada por defecto (apply_roi_augment=False). Los notebooks Fase 4 la activan en train.

Ejecutar desde la raíz del repo:
  python notebooks/v3_local/mejoras/scripts/patch_v3_notebook_fase4_augment_infrastructure.py
"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path("notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3.ipynb")

AUG_FN = '''

def apply_fase4_roi_geom_augment_uint8(
    image_crop_u8: np.ndarray,
    multiclass_crop: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Augmentación geométrica suave en crop ROI multiclase (solo train).
    Rotación ±4° y escala 0,98–1,02; 50% de no aplicar nada (radiografías).
    """
    if np.random.random() > 0.5:
        return image_crop_u8, multiclass_crop
    h, w = image_crop_u8.shape[:2]
    if h < 8 or w < 8:
        return image_crop_u8, multiclass_crop

    angle = float(np.random.uniform(-4.0, 4.0))
    scale = float(np.random.uniform(0.98, 1.02))
    nh = max(2, int(round(h * scale)))
    nw = max(2, int(round(w * scale)))

    img_p = Image.fromarray(image_crop_u8, mode="L")
    msk_u8 = multiclass_crop.astype(np.uint8, copy=False)
    msk_p = Image.fromarray(msk_u8, mode="L")
    img_p = img_p.resize((nw, nh), resample=Image.BILINEAR)
    msk_p = msk_p.resize((nw, nh), resample=Image.NEAREST)

    arr_i = np.array(img_p)
    arr_m = np.array(msk_p)
    hi, wi = arr_i.shape
    out_i = np.zeros((h, w), dtype=np.uint8)
    out_m = np.full((h, w), int(IGNORE_INDEX), dtype=np.uint8)
    if hi <= h and wi <= w:
        y0 = (h - hi) // 2
        x0 = (w - wi) // 2
        out_i[y0 : y0 + hi, x0 : x0 + wi] = arr_i
        out_m[y0 : y0 + hi, x0 : x0 + wi] = arr_m
    else:
        y0 = (hi - h) // 2
        x0 = (wi - w) // 2
        out_i = arr_i[y0 : y0 + h, x0 : x0 + w]
        out_m = arr_m[y0 : y0 + h, x0 : x0 + w]

    ri = Image.fromarray(out_i, mode="L")
    rm = Image.fromarray(out_m.astype(np.uint8), mode="L")
    ri2 = ri.rotate(angle, resample=Image.BILINEAR, fillcolor=0, expand=False)
    rm2 = rm.rotate(angle, resample=Image.NEAREST, fillcolor=int(IGNORE_INDEX), expand=False)
    out_i2 = np.array(ri2)
    out_m2 = np.array(rm2).astype(multiclass_crop.dtype, copy=False)
    return out_i2, out_m2


'''

OLD_PREP_SIG = """def prepare_multiclass_cascade_sample(
    row: pd.Series,
    output_size: tuple[int, int],
    roi_mode: str,
    roi_lookup: dict | None = None,
    apply_jitter: bool = False,
) -> dict:"""

NEW_PREP_SIG = """def prepare_multiclass_cascade_sample(
    row: pd.Series,
    output_size: tuple[int, int],
    roi_mode: str,
    roi_lookup: dict | None = None,
    apply_jitter: bool = False,
    apply_roi_augment: bool = False,
) -> dict:"""

OLD_CROP_BLOCK = """    image_crop = crop_array(image_raw, bbox)
    multiclass_crop = crop_array(multiclass_raw, bbox)

    image_crop = resize_image(image_crop, output_size).astype(np.float32) / 255.0"""

NEW_CROP_BLOCK = """    image_crop = crop_array(image_raw, bbox)
    multiclass_crop = crop_array(multiclass_raw, bbox)

    if apply_roi_augment:
        image_crop, multiclass_crop = apply_fase4_roi_geom_augment_uint8(image_crop, multiclass_crop)

    image_crop = resize_image(image_crop, output_size).astype(np.float32) / 255.0"""

OLD_DS_INIT = """    def __init__(
        self,
        table: pd.DataFrame,
        image_size: tuple[int, int],
        roi_mode: str,
        roi_lookup: dict | None = None,
        apply_jitter: bool = False,
    ):
        self.table = table.reset_index(drop=True).copy()
        self.image_size = image_size
        self.roi_mode = roi_mode
        self.roi_lookup = roi_lookup
        self.apply_jitter = apply_jitter"""

NEW_DS_INIT = """    def __init__(
        self,
        table: pd.DataFrame,
        image_size: tuple[int, int],
        roi_mode: str,
        roi_lookup: dict | None = None,
        apply_jitter: bool = False,
        apply_roi_augment: bool = False,
    ):
        self.table = table.reset_index(drop=True).copy()
        self.image_size = image_size
        self.roi_mode = roi_mode
        self.roi_lookup = roi_lookup
        self.apply_jitter = apply_jitter
        self.apply_roi_augment = apply_roi_augment"""

OLD_GETITEM_CALL = """        sample = prepare_multiclass_cascade_sample(
            row=row,
            output_size=self.image_size,
            roi_mode=self.roi_mode,
            roi_lookup=self.roi_lookup,
            apply_jitter=self.apply_jitter,
        )"""

NEW_GETITEM_CALL = """        sample = prepare_multiclass_cascade_sample(
            row=row,
            output_size=self.image_size,
            roi_mode=self.roi_mode,
            roi_lookup=self.roi_lookup,
            apply_jitter=self.apply_jitter,
            apply_roi_augment=self.apply_roi_augment,
        )"""

OLD_TRAIN_DS = """multiclass_train_ds = CascadedThoracolumbarDataset(
    multiclass_splits_df.query("partition == 'train'"),
    image_size=IMG_SIZE_MULTICLASS,
    roi_mode='gt_binary',
    roi_lookup=None,
    apply_jitter=True,
)"""

NEW_TRAIN_DS = """multiclass_train_ds = CascadedThoracolumbarDataset(
    multiclass_splits_df.query("partition == 'train'"),
    image_size=IMG_SIZE_MULTICLASS,
    roi_mode='gt_binary',
    roi_lookup=None,
    apply_jitter=True,
    apply_roi_augment=False,
)"""


def lines_from_str(s: str) -> list[str]:
    parts = s.splitlines(keepends=True)
    return parts if parts else [""]


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    hit_helpers = False
    hit_train = False
    for ci, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        s0 = "".join(cell["source"])
        s = s0
        if "def prepare_multiclass_cascade_sample" in s and "apply_fase4_roi_geom_augment_uint8" not in s:
            if OLD_PREP_SIG not in s:
                raise SystemExit("Firma prepare_multiclass no coincide")
            s = s.replace(OLD_PREP_SIG, NEW_PREP_SIG, 1)
            s = s.replace(OLD_CROP_BLOCK, NEW_CROP_BLOCK, 1)
            pos = s.find("def prepare_multiclass_cascade_sample")
            if pos == -1:
                raise SystemExit("prepare_multiclass no encontrado")
            s = s[:pos] + AUG_FN + s[pos:]
            s = s.replace(OLD_DS_INIT, NEW_DS_INIT, 1)
            s = s.replace(OLD_GETITEM_CALL, NEW_GETITEM_CALL, 1)
            hit_helpers = True
        if "multiclass_train_ds = CascadedThoracolumbarDataset" in s and "apply_roi_augment" not in s:
            if OLD_TRAIN_DS not in s:
                raise SystemExit("Bloque multiclass_train_ds no coincide")
            s = s.replace(OLD_TRAIN_DS, NEW_TRAIN_DS, 1)
            hit_train = True
        if s != s0:
            cell["source"] = lines_from_str(s)

    if not hit_helpers:
        raise SystemExit("No se parcheo celda helpers (¿ya aplicado?)")
    if not hit_train:
        raise SystemExit("No se parcheo multiclass_train_ds (¿ya aplicado?)")

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("OK:", NB)


if __name__ == "__main__":
    main()

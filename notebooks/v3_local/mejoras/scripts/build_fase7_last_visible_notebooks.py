# -*- coding: utf-8 -*-
"""
Genera notebooks Fase 7 — cascada V3 (baseline) + estimador last_visible + clipping.

**Autocontenido:** entrena binario + multiclase y luego el auxiliar last_visible en el mismo
notebook, como Fases 1–6. Fuente cascada: `train_spine_cascade_binary_to_thoracolumbar_v3.ipynb`.
Bloque last_visible adaptado desde `07_colab_train_last_visible_...ipynb`.

Desde la raiz del repo:
  python notebooks/v3_local/mejoras/scripts/build_fase7_last_visible_notebooks.py
"""
from __future__ import annotations

import copy
import json
import re
import uuid
from pathlib import Path
from typing import Any

from cascade_v3_mejora_notebook_common import (
    COLAB_CUDA_PROJECT_ROOT,
    append_execution_registry_cells,
    clear_code_cell_outputs,
    find_config_cell_index,
    lines_from_str,
    new_code_cell,
    new_markdown_cell,
    patch_config_cell_for_training_variant,
    patch_markdown_antes_de_ejecutar_output_dir,
    prepend_interpretacion_warning,
    prepend_mapa_and_optional_colab_cuda_cells,
    replace_cascade_checkpoint_paths,
)

SRC = Path(__file__).resolve().parents[2] / "train_spine_cascade_binary_to_thoracolumbar_v3.ipynb"
SRC_LAST = (
    Path(__file__).resolve().parents[3]
    / "07_colab_train_last_visible_estimator_and_clip_thoracolumbar_explained.ipynb"
)
DST_DIR = Path(__file__).resolve().parents[1] / (
    "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase7_auxiliares_rango_lastvis"
)

VARIANTS: dict[str, dict[str, str]] = {
    "cpu": {
        "suffix": "_cpu",
        "label": "CPU (perfil liviano)",
        "output_dir": "training_runs_cascade_v3_fase7_lastvis_cpu",
        "last_output_dir": "training_runs_last_visible_fase7_cpu",
        "binary_pt": "binary_spine_cascade_fase7_lastvis_cpu_best.pt",
        "multiclass_pt": "thoracolumbar_{MULTICLASS_SUBSET}_cascade_fase7_lastvis_cpu_best.pt",
        "last_visible_pt": "last_visible_estimator_fase7_lastvis_cpu_best.pt",
        "img_binary": "(320, 160)",
        "img_mc": "(320, 160)",
        "img_last": "(256, 128)",
        "batch": "2",
        "binary_epochs": "6",
        "multiclass_epochs": "8",
        "last_batch": "4",
        "last_epochs": "12",
        "nota": "Cascada liviana + last_visible; metricas exploratorias.",
    },
    "cuda": {
        "suffix": "_cuda",
        "label": "CUDA (perfil completo)",
        "output_dir": "training_runs_cascade_v3_fase7_lastvis_cuda",
        "last_output_dir": "training_runs_last_visible_fase7_cuda",
        "binary_pt": "binary_spine_cascade_fase7_lastvis_cuda_best.pt",
        "multiclass_pt": "thoracolumbar_{MULTICLASS_SUBSET}_cascade_fase7_lastvis_cuda_best.pt",
        "last_visible_pt": "last_visible_estimator_fase7_lastvis_cuda_best.pt",
        "img_binary": "(512, 256)",
        "img_mc": "(512, 256)",
        "img_last": "(384, 192)",
        "batch": "4",
        "binary_epochs": "14",
        "multiclass_epochs": "24",
        "last_batch": "8",
        "last_epochs": "40",
        "nota": "Cascada alineada al vivo + last_visible (referencia Colab).",
    },
}

FASE7_WORK_DF = r'''
# --- [FASE 7] Target last_visible_idx (mismo split que multiclass_splits_df) ---
canonical_labels = TARGET_LABELS
label_to_class_id = {label: class_names.index(label) for label in canonical_labels}


def extract_visible_range_from_mask(path: str) -> tuple[int | None, int | None, list[str]]:
    raw = np.array(Image.open(path), dtype=np.int32)
    labels_present: list[str] = []
    for rid in sorted(int(x) for x in np.unique(raw) if int(x) > 0):
        label = multiclass_map.get(rid)
        if label in canonical_labels:
            labels_present.append(label)
    if not labels_present:
        return None, None, []
    first_idx = canonical_labels.index(labels_present[0])
    last_idx = canonical_labels.index(labels_present[-1])
    return first_idx, last_idx, labels_present


work_df = multiclass_splits_df.copy()
first_last = work_df['multiclass_mask_path_abs'].apply(extract_visible_range_from_mask)
work_df['first_visible_idx'] = [item[0] for item in first_last]
work_df['last_visible_idx'] = [item[1] for item in first_last]
work_df['visible_labels_gt'] = [', '.join(item[2]) for item in first_last]
work_df = work_df.loc[
    work_df['first_visible_idx'].notna() & work_df['last_visible_idx'].notna()
].copy().reset_index(drop=True)
work_df['first_visible_idx'] = work_df['first_visible_idx'].astype(int)
work_df['last_visible_idx'] = work_df['last_visible_idx'].astype(int)
work_df['first_visible_label'] = work_df['first_visible_idx'].map(lambda idx: canonical_labels[int(idx)])
work_df['last_visible_label'] = work_df['last_visible_idx'].map(lambda idx: canonical_labels[int(idx)])

print('Muestras para last-visible estimator:', len(work_df))
display(work_df.groupby('partition').size().rename('images').reset_index())
display(work_df[['partition', 'split', 'image', 'first_visible_label', 'last_visible_label']].head(10))
'''

FASE7_HELPERS = r'''
# --- [FASE 7] Helpers adicionales (ROI / last_visible) ---


def normalize_image(image_2d: np.ndarray) -> np.ndarray:
    mean = float(image_2d.mean())
    std = float(image_2d.std())
    if std < 1e-6:
        return image_2d - mean
    return (image_2d - mean) / std


def build_coordinate_channels(height: int, width: int) -> np.ndarray:
    y_coords = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x_coords = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y_map = np.repeat(y_coords, width, axis=1)
    x_map = np.repeat(x_coords, height, axis=0)
    return np.stack([y_map, x_map], axis=0)
'''

FASE7_INFERENCE_PREP = r'''
# --- [FASE 7] Inferencia cascada (modelos de secciones 4-6) y tabla prep_df ---
binary_model.eval()
multiclass_model.eval()


def predict_binary_bbox_from_image(image_raw: np.ndarray) -> tuple[int, int, int, int] | None:
    image_resized = resize_image(image_raw, IMG_SIZE_BINARY).astype(np.float32) / 255.0
    image_tensor = torch.tensor(image_resized[None, None, ...], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits = binary_model(image_tensor)
        pred_mask_small = (torch.sigmoid(logits)[0, 0].detach().cpu().numpy() >= BINARY_THRESHOLD).astype(np.uint8)
    return bbox_from_mask(pred_mask_small, min_foreground_pixels=MIN_FOREGROUND_PIXELS)


def infer_multiclass_on_bbox(image_raw: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_crop = crop_array(image_raw, bbox)
    image_crop = resize_image(image_crop, IMG_SIZE_MULTICLASS).astype(np.float32) / 255.0
    image_tensor = torch.tensor(image_crop[None, None, ...], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits = multiclass_model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        if USE_MULTICLASS_TTA:
            flipped_input = torch.flip(image_tensor, dims=[3])
            flipped_logits = multiclass_model(flipped_input)
            flipped_probs = torch.softmax(flipped_logits, dim=1)
            flipped_probs = torch.flip(flipped_probs, dims=[3])
            probs = 0.5 * (probs + flipped_probs)
        probs_np = probs[0].detach().cpu().numpy().astype(np.float32)
        pred_mask = np.argmax(probs_np, axis=0).astype(np.int64)
    return image_crop, pred_mask, probs_np


def extract_aux_features_from_prediction(pred_mask: np.ndarray, prob_map: np.ndarray) -> np.ndarray:
    h, w = pred_mask.shape
    fg_mask = (pred_mask > 0).astype(np.float32)
    total_fg = float(fg_mask.sum()) + 1e-6

    presence = []
    area_ratio = []
    centroid_y = []
    y_min_norm = []
    y_max_norm = []
    height_span_norm = []
    mean_confidence = []
    for label in canonical_labels:
        class_id = class_names.index(label)
        class_mask = pred_mask == class_id
        area = float(class_mask.sum())
        presence.append(1.0 if area >= PRESENCE_THRESHOLD_PIXELS else 0.0)
        area_ratio.append(area / total_fg)
        if area > 0:
            ys, _ = np.where(class_mask)
            centroid_y.append(float(np.mean(ys) / max(h - 1, 1)))
            y_min_norm.append(float(np.min(ys) / max(h - 1, 1)))
            y_max_norm.append(float(np.max(ys) / max(h - 1, 1)))
            height_span_norm.append(float((np.max(ys) - np.min(ys) + 1) / max(h, 1)))
            mean_confidence.append(float(prob_map[class_id][class_mask].mean()))
        else:
            centroid_y.append(0.0)
            y_min_norm.append(0.0)
            y_max_norm.append(0.0)
            height_span_norm.append(0.0)
            mean_confidence.append(0.0)

    pred_present_indices = [i for i, p in enumerate(presence) if p > 0.5]
    min_present_idx = float(min(pred_present_indices)) if pred_present_indices else 0.0
    max_present_idx = float(max(pred_present_indices)) if pred_present_indices else 0.0
    num_present = float(len(pred_present_indices))

    row_profile = fg_mask.sum(axis=1).astype(np.float32)
    if row_profile.max() > 0:
        row_profile = row_profile / row_profile.max()
    binned_profile = np.array_split(row_profile, PROFILE_BINS)
    profile_features = [float(chunk.mean()) for chunk in binned_profile]

    return np.array(
        presence
        + area_ratio
        + centroid_y
        + y_min_norm
        + y_max_norm
        + height_span_norm
        + mean_confidence
        + [
            min_present_idx / (len(canonical_labels) - 1),
            max_present_idx / (len(canonical_labels) - 1),
            num_present / len(canonical_labels),
        ]
        + profile_features,
        dtype=np.float32,
    )


def estimate_last_visible_from_mask(pred_mask: np.ndarray) -> int:
    present_indices = [
        canonical_labels.index(class_names[int(class_id)])
        for class_id in sorted(int(x) for x in np.unique(pred_mask) if int(x) > 0)
        if int(class_id) < len(class_names) and class_names[int(class_id)] in canonical_labels
    ]
    if not present_indices:
        return 0
    return int(max(present_indices))


prep_rows = []
image_crop_lookup = {}
raw_pred_lookup_all = {}
target_lookup_all = {}

for _, row in work_df.iterrows():
    image_raw = read_gray(row['radiograph_path_abs'])
    image_shape = image_raw.shape

    if row['partition'] == 'train':
        gt_binary = build_binary_mask(row['binary_mask_path_abs'], size=None)
        bbox = bbox_from_mask(gt_binary, min_foreground_pixels=MIN_FOREGROUND_PIXELS)
        roi_source = 'gt_binary'
    else:
        bbox_small = predict_binary_bbox_from_image(image_raw)
        bbox = scale_bbox(bbox_small, src_shape=IMG_SIZE_BINARY, dst_shape=image_shape) if bbox_small is not None else None
        roi_source = 'pred_binary'

    if bbox is None:
        x0, y0, x1, y1 = 0, 0, image_shape[1], image_shape[0]
        roi_source = f'{roi_source}_fallback_full_image'
    else:
        x0, y0, x1, y1 = expand_bbox(bbox, image_shape=image_shape, pad_x=ROI_PAD_X, pad_y=ROI_PAD_Y)

    bbox_final = (int(x0), int(y0), int(x1), int(y1))
    image_crop_2d, raw_pred_mask, raw_prob_map = infer_multiclass_on_bbox(image_raw, bbox_final)
    aux_features = extract_aux_features_from_prediction(raw_pred_mask, raw_prob_map)
    heuristic_last_idx = estimate_last_visible_from_mask(raw_pred_mask)

    target_full = build_multiclass_mask(row['multiclass_mask_path_abs'], size=None)
    target_crop = crop_array(target_full, bbox_final)
    target_crop = resize_mask(target_crop, IMG_SIZE_MULTICLASS).astype(np.int64)

    prep_rows.append({
        'unique_sample_id': row['unique_sample_id'],
        'partition': row['partition'],
        'split': row['split'],
        'image': row['image'],
        'radiograph_path_abs': row['radiograph_path_abs'],
        'multiclass_mask_path_abs': row['multiclass_mask_path_abs'],
        'first_visible_idx': int(row['first_visible_idx']),
        'last_visible_idx': int(row['last_visible_idx']),
        'first_visible_label': row['first_visible_label'],
        'last_visible_label': row['last_visible_label'],
        'heuristic_last_idx': heuristic_last_idx,
        'bbox_x0': bbox_final[0],
        'bbox_y0': bbox_final[1],
        'bbox_x1': bbox_final[2],
        'bbox_y1': bbox_final[3],
        'roi_source': roi_source,
        'aux_features': aux_features,
    })

    image_crop_lookup[row['unique_sample_id']] = image_crop_2d
    raw_pred_lookup_all[row['unique_sample_id']] = raw_pred_mask
    target_lookup_all[row['unique_sample_id']] = target_crop

prep_df = pd.DataFrame(prep_rows)
aux_feature_matrix = np.stack([row['aux_features'] for row in prep_rows], axis=0)
aux_feature_dim = int(aux_feature_matrix.shape[1])

print('prep_df:', prep_df.shape)
print('aux_feature_dim:', aux_feature_dim)
display(prep_df.groupby(['partition', 'roi_source']).size().rename('images').reset_index())
'''


def _fase7_config_snippet(cfg: dict[str, str]) -> str:
    return f"""
# --- [FASE 7] Estimador last_visible + clipping (salidas en LAST_OUTPUT_DIR) ---
LAST_OUTPUT_DIR = ROOT / 'outputs' / 'analysis_outputs_v3' / '{cfg["last_output_dir"]}'
LAST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_SIZE_LAST = {cfg["img_last"]}
LAST_BATCH_SIZE = {cfg["last_batch"]}
LAST_NUM_WORKERS = 0
LAST_EPOCHS = {cfg["last_epochs"]}
LAST_LR = 6e-4
LAST_WEIGHT_DECAY = 1e-4
LAST_PATIENCE = 8
LAST_DROPOUT = 0.25
LAST_LABEL_SMOOTHING = 0.04
LAST_EXPECTATION_LOSS_WEIGHT = 0.20
LAST_GRAD_CLIP_NORM = 1.0
LAST_HEURISTIC_BLEND = 0.20
LAST_EXPECTATION_BLEND = 0.30
USE_MULTICLASS_TTA = True
USE_AMP = DEVICE.type == "cuda"
PRESENCE_THRESHOLD_PIXELS = 40
PROFILE_BINS = 24
N_VIS_SAMPLES = 8
LAST_VISIBLE_MODEL_PATH = MODEL_DIR / '{cfg["last_visible_pt"]}'
# Comparacion opcional con estimador visible-range del notebook 07 (si existe en disco)
PREV_RANGE_TEST_PATH = (
    ROOT / 'reports' / 'analysis_outputs' / 'visible_range_estimator_thoracolumbar_explained'
    / 'visible_range_test_predictions.csv'
)
"""


def _mapa_md(variant_key: str, cfg: dict[str, str]) -> str:
    colab = (
        "\n### Colab (`_cuda`)\n\n"
        "Montar Drive y ejecutar el notebook completo (cascada + last_visible). "
        f"Raíz típica: `{COLAB_CUDA_PROJECT_ROOT.as_posix()}`.\n"
        if variant_key == "cuda"
        else ""
    )
    return f"""# Fase 7 — Cascada + last_visible + clipping (`{cfg["suffix"]}`)

## Finalidad

Notebook **autocontenido** (como Fases 1–6): entrena **binario + multiclase** desde el baseline V3 vivo y, a continuación, el estimador **`last_visible_idx`** y las comparaciones de **clipping**.

No depende de ejecutar otros notebooks `mejorafase*`. Tras adoptar mejoras previas, este script se regenera desde `train_spine_cascade_binary_to_thoracolumbar_v3.ipynb`.

## Mapa de cambios

| Bloque | Qué hace |
|--------|----------|
| Secciones 1–7 (base) | Mismo pipeline cascada; `OUTPUT_DIR` → `{cfg["output_dir"]}`; checkpoints `{cfg["binary_pt"]}` / multiclase fase7. |
| **Sección 8 (Fase 7)** | Target `last_visible`, prep con cascada **de este run**, entrena `LastVisibleEstimator`, clipping y CSV en `{cfg["last_output_dir"]}`. |

**Variante:** {cfg["label"]} — {cfg["nota"]}
{colab}
"""


def _extract_last_visible_cells(nb07: dict) -> list[dict]:
    cells: list[dict] = []
    started = False
    for c in nb07["cells"]:
        if c.get("cell_type") == "markdown":
            src = "".join(c.get("source", []))
            if "## 2. Metadata" in src:
                started = True
        if started:
            cells.append(copy.deepcopy(c))
    if not cells:
        raise RuntimeError("No se encontraron celdas last_visible en notebook 07 (desde seccion 2)")
    return cells


def _adapt_last_visible_source(src: str, cfg: dict[str, str]) -> str:
    """Adapta celdas del 07 al esquema del baseline V3."""
    src = src.replace(
        "scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)",
        "scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)",
    )
    src = src.replace(
        "with torch.cuda.amp.autocast(enabled=USE_AMP):",
        "with torch.amp.autocast('cuda', enabled=USE_AMP):",
    )
    src = src.replace("PROJECT_DIR", "ROOT")
    src = src.replace("OUTPUT_DIR", "LAST_OUTPUT_DIR")
    src = src.replace(
        'MODEL_DIR / "last_visible_estimator_thoracolumbar_best.pt"',
        f"MODEL_DIR / '{cfg['last_visible_pt']}'",
    )
    src = src.replace(
        "MODEL_DIR / 'last_visible_estimator_thoracolumbar_best.pt'",
        f"MODEL_DIR / '{cfg['last_visible_pt']}'",
    )
    src = re.sub(
        r"best_model_path = MODEL_DIR / [^\n]+",
        f"best_model_path = MODEL_DIR / '{cfg['last_visible_pt']}'",
        src,
        count=1,
    )
    src = src.replace('TARGET_SUBSET = "partial"', 'TARGET_SUBSET = MULTICLASS_SUBSET')
    src = src.replace("subset_flag = 'usable_for_thoracolumbar_core' if TARGET_SUBSET == 'core'", "")
    src = src.replace("subset_flag = 'usable_for_thoracolumbar_partial'", "")
    # Baseline: MULTICLASS_SUBSET; notebook 07: TARGET_SUBSET (no tocar alias)
    src = re.sub(r"(?<!MULTICLASS_)TARGET_SUBSET", "MULTICLASS_SUBSET", src)
    return src


def _prepare_last_visible_cells(raw_cells: list[dict], cfg: dict[str, str]) -> list[dict]:
    out: list[dict] = []
    i = 0
    while i < len(raw_cells):
        c = raw_cells[i]
        if c.get("cell_type") == "markdown":
            title = "".join(c.get("source", []))
            if "## 2. Metadata" in title:
                out.append(
                    new_markdown_cell(
                        "## Seccion 8. Fase 7 — Metadata y target `last_visible_idx`\n\n"
                        "Target supervisado a partir de mascaras GT; mismo split que `multiclass_splits_df`.\n"
                    )
                )
                out.append(new_code_cell(FASE7_WORK_DF.strip()))
                i += 2
                continue
            if "## 3. Distribucion" in title:
                out.append(
                    new_markdown_cell(
                        "### [FASE 7] Distribucion del target `last_visible_idx`\n\n"
                        + title.split("\n", 1)[-1]
                    )
                )
                i += 1
                continue
            if "## 4. Utilidades" in title:
                out.append(
                    new_markdown_cell(
                        "### [FASE 7] Utilidades ROI / features auxiliares\n\n"
                        "Reutiliza helpers del baseline; anade normalizacion y canales de coordenadas.\n"
                    )
                )
                out.append(new_code_cell(FASE7_HELPERS.strip()))
                i += 2
                continue
            if "## 5. Construccion de ROI" in title:
                out.append(
                    new_markdown_cell(
                        "### [FASE 7] ROI e inferencia multiclase (cascada de este notebook)\n\n"
                        "Usa `binary_model` y `multiclass_model` entrenados arriba (no checkpoints de otras fases).\n"
                    )
                )
                out.append(new_code_cell(FASE7_INFERENCE_PREP.strip()))
                i += 2
                continue
            if title.startswith("## "):
                title = title.replace("## ", "### [FASE 7] ", 1)
                c = copy.deepcopy(c)
                c["source"] = lines_from_str(title)
        if c.get("cell_type") == "code":
            src = _adapt_last_visible_source("".join(c.get("source", [])), cfg)
            c = new_code_cell(src)
        else:
            c = copy.deepcopy(c)
            if c.get("cell_type") == "markdown" and "id" not in c:
                c["id"] = uuid.uuid4().hex[:8]
        out.append(c)
        i += 1
    return out


def _insert_before_interpretacion(nb: dict, new_cells: list[dict]) -> None:
    idx = None
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "markdown":
            continue
        if "## Como interpretar" in "".join(c.get("source", [])):
            idx = i
            break
    if idx is None:
        raise RuntimeError("No se encontro seccion 'Como interpretar' en el baseline")
    nb["cells"][idx:idx] = new_cells


def _ensure_fase7_imports(nb: dict) -> None:
    ic = find_config_cell_index(nb)
    s = "".join(nb["cells"][ic]["source"])
    if "import copy\n" not in s:
        if "import json\n" in s:
            s = s.replace("import json\n", "import copy\nimport json\n", 1)
        else:
            s = "import copy\n" + s
        nb["cells"][ic]["source"] = lines_from_str(s)


def _ensure_fase7_aliases(nb: dict) -> None:
    ic = find_config_cell_index(nb)
    s = "".join(nb["cells"][ic]["source"])
    if "TARGET_SUBSET = MULTICLASS_SUBSET" in s:
        return
    if "MULTICLASS_SUBSET = 'core'" in s:
        s = s.replace(
            "MULTICLASS_SUBSET = 'core'",
            "MULTICLASS_SUBSET = 'core'   # cambiar a 'partial' en una iteracion posterior\n"
            "TARGET_SUBSET = MULTICLASS_SUBSET  # alias notebook 07\n",
            1,
        )
        nb["cells"][ic]["source"] = lines_from_str(s)


def _ensure_fase7_config_paths(nb: dict) -> None:
    """Constantes del bloque 07 que no estan en el baseline pero las usan celdas Fase 7."""
    ic = find_config_cell_index(nb)
    s = "".join(nb["cells"][ic]["source"])
    if "PREV_RANGE_TEST_PATH" in s:
        return
    anchor = f"LAST_VISIBLE_MODEL_PATH = MODEL_DIR / 'last_visible_estimator_fase7"
    if anchor not in s and "LAST_VISIBLE_MODEL_PATH" not in s:
        raise RuntimeError("LAST_VISIBLE_MODEL_PATH no encontrado para anclar PREV_RANGE_TEST_PATH")
    insert = (
        "\n# Comparacion opcional con visible-range del notebook 07 (si el CSV existe)\n"
        "PREV_RANGE_TEST_PATH = (\n"
        "    ROOT / 'reports' / 'analysis_outputs' / 'visible_range_estimator_thoracolumbar_explained'\n"
        "    / 'visible_range_test_predictions.csv'\n"
        ")\n"
    )
    if "LAST_VISIBLE_MODEL_PATH" in s:
        lines = s.splitlines(keepends=True)
        out: list[str] = []
        for line in lines:
            out.append(line)
            if line.strip().startswith("LAST_VISIBLE_MODEL_PATH ="):
                out.append(insert)
        s = "".join(out)
    nb["cells"][ic]["source"] = lines_from_str(s)


def _patch_last_visible_training_cells(nb: dict, cfg: dict[str, str]) -> None:
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        s = "".join(c.get("source", []))
        if "expectation_loss" not in s or "GradScaler" not in s:
            continue
        s2 = _adapt_last_visible_source(s, cfg)
        if s2 != s:
            c["source"] = lines_from_str(s2)


def _patch_config_fase7(nb: dict, cfg: dict[str, str]) -> None:
    _ensure_fase7_imports(nb)
    _ensure_fase7_aliases(nb)
    ic = find_config_cell_index(nb)
    s = "".join(nb["cells"][ic]["source"])
    if "LAST_OUTPUT_DIR" not in s:
        m = re.search(r"^MULTICLASS_EPOCHS = \d+\s*$", s, re.MULTILINE)
        if not m:
            raise RuntimeError("MULTICLASS_EPOCHS no encontrado en config")
        insert_at = m.end() + 1
        s = s[:insert_at] + _fase7_config_snippet(cfg) + s[insert_at:]
        nb["cells"][ic]["source"] = lines_from_str(s)
    _ensure_fase7_config_paths(nb)


def build_notebook(variant_key: str) -> Path:
    cfg = VARIANTS[variant_key].copy()
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    nb07 = json.loads(SRC_LAST.read_text(encoding="utf-8"))

    mapa = _mapa_md(variant_key, cfg)
    if variant_key == "cuda":
        mapa += (
            "\n## Colab — raíz por defecto\n\n"
            f"`{COLAB_CUDA_PROJECT_ROOT.as_posix()}`\n\n---\n\n"
        )
    prepend_mapa_and_optional_colab_cuda_cells(nb, variant_key, mapa)
    patch_config_cell_for_training_variant(nb, cfg, variant_key, "FASE 7 cascada+lastvis")
    _patch_config_fase7(nb, cfg)
    replace_cascade_checkpoint_paths(nb, cfg)
    patch_markdown_antes_de_ejecutar_output_dir(nb, cfg["output_dir"])

  # Bloque last_visible antes de interpretacion
    raw_last = _extract_last_visible_cells(nb07)
    last_cells = _prepare_last_visible_cells(raw_last, cfg)
    resume = [
        new_markdown_cell(
            "### [FASE 7] Reanudar tras parches del notebook\n\n"
            "Si ya corriste cascada + last_visible: (1) celda **config**, (2) desde la celda que falló. "
            "No repitas entrenamiento salvo reinicio de kernel.\n"
        ),
        new_code_cell("TARGET_SUBSET = MULTICLASS_SUBSET  # alias notebook 07\n"),
    ]
    _insert_before_interpretacion(nb, resume + last_cells)
    _patch_last_visible_training_cells(nb, cfg)
    _ensure_fase7_imports(nb)
    _ensure_fase7_config_paths(nb)

    prepend_interpretacion_warning(nb)
    append_execution_registry_cells(nb)
    clear_code_cell_outputs(nb)

    fname = (
        "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase7_auxiliares_rango_lastvis"
        f"_{variant_key}.ipynb"
    )
    out = DST_DIR / fname
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    if not SRC_LAST.exists():
        raise FileNotFoundError(SRC_LAST)
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for key in ("cpu", "cuda"):
        p = build_notebook(key)
        print("Escrito:", p)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Genera notebooks Fase 6 — post-proceso ligero en máscaras multiclase (inferencia en test).

A partir del cascada V3 vivo (Fase 3 + 4): mismo entrenamiento; en **test** se reportan métricas
**sin post** (como el base) y **con post** Fase 6 (islas pequeñas por clase; opcional mediana vertical).

Desde la raíz del repo:
  python notebooks/v3_local/mejoras/scripts/build_fase6_postproceso_notebooks.py
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
DST_DIR = Path(__file__).resolve().parents[1] / "train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase6_postproceso_ligero"

VARIANTS: dict[str, dict[str, str]] = {
    "cpu": {
        "label": "CPU (perfil liviano)",
        "output_dir": "training_runs_cascade_v3_fase6_postproc_cpu",
        "binary_pt": "binary_spine_cascade_fase6_postproc_cpu_best.pt",
        "multiclass_pt": "thoracolumbar_{MULTICLASS_SUBSET}_cascade_fase6_postproc_cpu_best.pt",
        "img_binary": "(320, 160)",
        "img_mc": "(320, 160)",
        "batch": "2",
        "binary_epochs": "6",
        "multiclass_epochs": "8",
        "nota": "Perfil liviano; post Fase 6 solo en evaluación test (islas).",
    },
    "cuda": {
        "label": "CUDA (perfil completo)",
        "output_dir": "training_runs_cascade_v3_fase6_postproc_cuda",
        "binary_pt": "binary_spine_cascade_fase6_postproc_cuda_best.pt",
        "multiclass_pt": "thoracolumbar_{MULTICLASS_SUBSET}_cascade_fase6_postproc_cuda_best.pt",
        "img_binary": "(512, 256)",
        "img_mc": "(512, 256)",
        "batch": "4",
        "binary_epochs": "14",
        "multiclass_epochs": "24",
        "nota": "Misma escala y épocas que el vivo. Post-proceso Fase 6 solo al cerrar test multiclase.",
    },
}

OLD_EVAL_MC_SIGNATURE = (
    "def evaluate_multiclass(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) "
    "-> tuple[dict[str, float], pd.DataFrame]:"
)

FASE6_HELPERS_AND_EVAL = r'''

def _fase6_bfs_component_on_mask(m: np.ndarray, sy: int, sx: int, visited: np.ndarray) -> list[tuple[int, int]]:
    """Componente 4-vecinos dentro de la mascara booleana m (True = pixel de interes)."""
    h, w = m.shape
    stack = [(sy, sx)]
    comp: list[tuple[int, int]] = []
    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w or not m[y, x] or visited[y, x]:
            continue
        visited[y, x] = True
        comp.append((y, x))
        stack.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])
    return comp


def apply_fase6_postprocess_multiclass_pred(
    pred_hw: np.ndarray,
    *,
    num_classes: int,
    min_island_pixels: int,
    vertical_median_kernel: int = 0,
) -> np.ndarray:
    """Fase 6: elimina islas pequeñas por etiqueta (4-vecinos). Opcional: mediana vertical (kernel impar >=3)."""
    out = pred_hw.astype(np.int32, copy=True)
    h, w = out.shape

    for c in range(1, num_classes):
        m = out == c
        visited = np.zeros((h, w), dtype=bool)
        for y in range(h):
            for x in range(w):
                if not m[y, x] or visited[y, x]:
                    continue
                comp = _fase6_bfs_component_on_mask(m, y, x, visited)
                if len(comp) < min_island_pixels:
                    for yy, xx in comp:
                        out[yy, xx] = 0

    k = int(vertical_median_kernel)
    if k >= 3 and k % 2 == 1:
        half = k // 2
        smoothed = out.copy()
        for y in range(h):
            for x in range(w):
                sl = out[max(0, y - half) : min(h, y + half + 1), x]
                vals, cnts = np.unique(sl, return_counts=True)
                smoothed[y, x] = int(vals[int(np.argmax(cnts))])
        out = smoothed

    return out.astype(pred_hw.dtype, copy=False)


def evaluate_multiclass(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    *,
    apply_fase6_postprocess: bool = False,
    fase6_min_island_pixels: int = 64,
    fase6_vertical_median_k: int = 0,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    correct = 0.0
    total_valid_pixels = 0.0
    intersection = np.zeros(num_classes, dtype=np.float64)
    pred_area = np.zeros(num_classes, dtype=np.float64)
    target_area = np.zeros(num_classes, dtype=np.float64)

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(DEVICE)
            targets = batch['mask'].to(DEVICE)
            logits = model(images)
            loss = loss_fn(logits, targets) + dice_loss_multiclass(
                logits, targets, num_classes=num_classes, ignore_index=IGNORE_INDEX
            )
            total_loss += float(loss.item())
            total_batches += 1

            preds = torch.argmax(logits, dim=1)
            if apply_fase6_postprocess:
                preds_np = preds.detach().cpu().numpy()
                for bi in range(preds_np.shape[0]):
                    preds_np[bi] = apply_fase6_postprocess_multiclass_pred(
                        preds_np[bi],
                        num_classes=num_classes,
                        min_island_pixels=fase6_min_island_pixels,
                        vertical_median_kernel=fase6_vertical_median_k,
                    )
                preds = torch.from_numpy(preds_np).to(device=preds.device, dtype=preds.dtype)

            valid = targets != IGNORE_INDEX
            correct += float(((preds == targets) & valid).sum().item())
            total_valid_pixels += float(valid.sum().item())

            preds_np = preds[valid].detach().cpu().numpy()
            targets_np = targets[valid].detach().cpu().numpy()
            for class_idx in range(num_classes):
                pred_mask = preds_np == class_idx
                target_mask = targets_np == class_idx
                intersection[class_idx] += np.logical_and(pred_mask, target_mask).sum()
                pred_area[class_idx] += pred_mask.sum()
                target_area[class_idx] += target_mask.sum()

    dice = (2.0 * intersection + 1e-6) / (pred_area + target_area + 1e-6)
    iou = (intersection + 1e-6) / (pred_area + target_area - intersection + 1e-6)
    per_class_df = pd.DataFrame(
        {
            'class_id': np.arange(num_classes),
            'class_name': class_names,
            'pred_pixels': pred_area,
            'target_pixels': target_area,
            'dice': dice,
            'iou': iou,
        }
    )
    fg_df = per_class_df.loc[per_class_df['class_id'] > 0].copy()
    metrics = {
        'loss': total_loss / max(total_batches, 1),
        'pixel_accuracy': (correct + 1e-6) / (total_valid_pixels + 1e-6),
        'macro_dice_fg': float(fg_df['dice'].mean()),
        'macro_iou_fg': float(fg_df['iou'].mean()),
    }
    return metrics, per_class_df


'''

OLD_TEST_BLOCK = """multiclass_test_metrics, multiclass_per_class_df = evaluate_multiclass(
    multiclass_model,
    multiclass_test_loader,
    loss_fn=multiclass_loss_fn,
)
multiclass_elapsed_min = (time.time() - multiclass_start) / 60.0

multiclass_model_path = MODEL_DIR / f'thoracolumbar_{MULTICLASS_SUBSET}_cascade_best.pt'
multiclass_history_path = OUTPUT_DIR / f'thoracolumbar_{MULTICLASS_SUBSET}_history.csv'
multiclass_test_metrics_path = OUTPUT_DIR / f'thoracolumbar_{MULTICLASS_SUBSET}_test_metrics.csv'
multiclass_per_class_path = OUTPUT_DIR / f'thoracolumbar_{MULTICLASS_SUBSET}_per_class_metrics.csv'
multiclass_class_weights_path = OUTPUT_DIR / f'thoracolumbar_{MULTICLASS_SUBSET}_class_weights.csv'

torch.save(multiclass_model.state_dict(), multiclass_model_path)
multiclass_history_df.to_csv(multiclass_history_path, index=False)
pd.DataFrame([multiclass_test_metrics]).to_csv(multiclass_test_metrics_path, index=False)
multiclass_per_class_df.to_csv(multiclass_per_class_path, index=False)
class_weights_df.to_csv(multiclass_class_weights_path, index=False)

print('\\nMejor val_macro_dice_fg:', round(best_multiclass_val_macro_dice, 4))
print('Metricas finales en test:', multiclass_test_metrics)
print('Tiempo de entrenamiento multiclase (min):', round(multiclass_elapsed_min, 2))

display(class_weights_df)
display(multiclass_history_df)
display(pd.DataFrame([multiclass_test_metrics]))
display(multiclass_per_class_df.sort_values('dice', ascending=False))
plot_history(multiclass_history_df, f'Historia de entrenamiento: cascada thoracolumbar {MULTICLASS_SUBSET}')"""

NEW_TEST_BLOCK = """multiclass_test_metrics, multiclass_per_class_df = evaluate_multiclass(
    multiclass_model,
    multiclass_test_loader,
    loss_fn=multiclass_loss_fn,
)
multiclass_test_metrics_post, multiclass_per_class_df_post = evaluate_multiclass(
    multiclass_model,
    multiclass_test_loader,
    loss_fn=multiclass_loss_fn,
    apply_fase6_postprocess=True,
    fase6_min_island_pixels=FASE6_MIN_ISLAND_PIXELS,
    fase6_vertical_median_k=FASE6_VERTICAL_MEDIAN_K,
)
multiclass_elapsed_min = (time.time() - multiclass_start) / 60.0

multiclass_model_path = MODEL_DIR / f'thoracolumbar_{MULTICLASS_SUBSET}_cascade_best.pt'
multiclass_history_path = OUTPUT_DIR / f'thoracolumbar_{MULTICLASS_SUBSET}_history.csv'
multiclass_test_metrics_path = OUTPUT_DIR / f'thoracolumbar_{MULTICLASS_SUBSET}_test_metrics.csv'
multiclass_per_class_path = OUTPUT_DIR / f'thoracolumbar_{MULTICLASS_SUBSET}_per_class_metrics.csv'
multiclass_test_metrics_post_path = OUTPUT_DIR / f'thoracolumbar_{MULTICLASS_SUBSET}_test_metrics_fase6_post.csv'
multiclass_per_class_post_path = OUTPUT_DIR / f'thoracolumbar_{MULTICLASS_SUBSET}_per_class_metrics_fase6_post.csv'
multiclass_class_weights_path = OUTPUT_DIR / f'thoracolumbar_{MULTICLASS_SUBSET}_class_weights.csv'

torch.save(multiclass_model.state_dict(), multiclass_model_path)
multiclass_history_df.to_csv(multiclass_history_path, index=False)
pd.DataFrame([multiclass_test_metrics]).to_csv(multiclass_test_metrics_path, index=False)
multiclass_per_class_df.to_csv(multiclass_per_class_path, index=False)
pd.DataFrame([multiclass_test_metrics_post]).to_csv(multiclass_test_metrics_post_path, index=False)
multiclass_per_class_df_post.to_csv(multiclass_per_class_post_path, index=False)
class_weights_df.to_csv(multiclass_class_weights_path, index=False)

print('\\nMejor val_macro_dice_fg:', round(best_multiclass_val_macro_dice, 4))
print('Metricas finales en test (sin post Fase 6):', multiclass_test_metrics)
print('Metricas finales en test (con post Fase 6):', multiclass_test_metrics_post)
print('Tiempo de entrenamiento multiclase (min):', round(multiclass_elapsed_min, 2))

display(class_weights_df)
display(multiclass_history_df)
display(pd.DataFrame([multiclass_test_metrics]))
display(pd.DataFrame([multiclass_test_metrics_post]))
display(multiclass_per_class_df.sort_values('dice', ascending=False))
display(multiclass_per_class_df_post.sort_values('dice', ascending=False))
plot_history(multiclass_history_df, f'Historia de entrenamiento: cascada thoracolumbar {MULTICLASS_SUBSET}')"""

FASE6_CONFIG_SNIPPET = """
# --- Fase 6: post-proceso en test multiclase (islas minimas por clase; opcional mediana vertical) ---
FASE6_MIN_ISLAND_PIXELS = 64
FASE6_VERTICAL_MEDIAN_K = 0  # p. ej. 3 para suavizado vertical ligero (impar); 0 = desactivado
"""


def _patch_evaluate_multiclass_cell(nb: dict) -> None:
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s = "".join(cell["source"])
        if OLD_EVAL_MC_SIGNATURE not in s or "apply_fase6_postprocess_multiclass_pred" in s:
            continue
        start = s.find(OLD_EVAL_MC_SIGNATURE)
        end = s.find("\n\ndef plot_history", start)
        if start == -1 or end == -1:
            raise RuntimeError("evaluate_multiclass: no se encontro bloque o plot_history")
        new_src = s[:start] + FASE6_HELPERS_AND_EVAL.strip() + "\n" + s[end:]
        cell["source"] = lines_from_str(new_src)
        return
    raise RuntimeError("Celda con evaluate_multiclass no encontrada")


def _patch_config_fase6_constants(nb: dict) -> None:
    ic = find_config_cell_index(nb)
    s = "".join(nb["cells"][ic]["source"])
    if "FASE6_MIN_ISLAND_PIXELS" in s:
        return
    anchor = "MULTICLASS_EPOCHS = 24\n"
    if anchor not in s:
        # perfil CPU puede tener otro valor; buscar linea MULTICLASS_EPOCHS
        import re

        m = re.search(r"^MULTICLASS_EPOCHS = \d+\s*$", s, re.MULTILINE)
        if not m:
            raise RuntimeError("MULTICLASS_EPOCHS no encontrado en config")
        anchor = m.group(0) + "\n"
    if anchor not in s:
        raise RuntimeError("ancla MULTICLASS_EPOCHS")
    s = s.replace(anchor, anchor + FASE6_CONFIG_SNIPPET.lstrip("\n"), 1)
    nb["cells"][ic]["source"] = lines_from_str(s)


def _patch_multiclass_test_cell(nb: dict) -> None:
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s = "".join(cell["source"])
        if "multiclass_test_metrics_post" in s:
            return
        if OLD_TEST_BLOCK in s:
            cell["source"] = lines_from_str(s.replace(OLD_TEST_BLOCK, NEW_TEST_BLOCK, 1))
            return
    raise RuntimeError("Bloque de test multiclase esperado no encontrado")


def build_notebook(variant_key: str) -> Path:
    cfg = VARIANTS[variant_key].copy()
    nb = json.loads(SRC.read_text(encoding="utf-8"))

    mapa_md = f"""# Fase 6 — Post-proceso ligero anatómico (`_{variant_key}`)

## Finalidad

Según el plan (`PLAN_ACCION_AJUSTES_MODELOS.md` §3 Fase 6), aplicar un **post-proceso ligero** sobre las predicciones multiclase en **test** para reducir islas espurias por etiqueta (y opcionalmente un suavizado vertical muy local), **sin** sustituir el pipeline completo del informe.

## Mapa de cambios (vs notebook vivo)

| Ubicación | Qué cambia |
|-----------|------------|
| Config | `FASE6_MIN_ISLAND_PIXELS`, `FASE6_VERTICAL_MEDIAN_K` (mediana 0 por defecto). |
| `evaluate_multiclass` | Acepta `apply_fase6_postprocess`; BFS por clase para eliminar componentes menores que el umbral en píxeles. |
| Cierre multiclase | Evalúa test **dos veces**: métricas sin post (CSV estándar) y con post (`*_test_metrics_fase6_post.csv`, `*_per_class_metrics_fase6_post.csv`). |
| Entrenamiento binario / multiclase | Igual que el vivo; checkpoints y `OUTPUT_DIR` propios de esta fase. |

## Variante: **{cfg['label']}**

{cfg['nota']}

---

"""
    mapa_md += (
        "## Colab — raíz por defecto (`_cuda`)\n\n"
        "Si el repo llega a Drive como **Other computers / Mi portátil / ScoliosisSegmentation**, "
        "la celda de configuración prueba **antes** que `MyDrive` la ruta:\n\n"
        f"`{COLAB_CUDA_PROJECT_ROOT.as_posix()}`\n\n"
        "---\n\n"
    )
    prepend_mapa_and_optional_colab_cuda_cells(nb, variant_key, mapa_md)

    insert_markdown_before_read_gray(
        nb,
        lines_from_str(
            "### [FASE 6 — POST-PROCESO] Islas mínimas por etiqueta en test multiclase\n\n"
            "- El entrenamiento y la selección por `val_macro_dice_fg` son **idénticos** al vivo.\n"
            "- Tras el entrenamiento, el **test** se reporta con y sin post-proceso Fase 6 (ver CSV `*_fase6_post`).\n"
            "- Ajustar `FASE6_MIN_ISLAND_PIXELS` / `FASE6_VERTICAL_MEDIAN_K` en la celda de configuración si se itera.\n"
        ),
    )

    patch_config_cell_for_training_variant(nb, cfg, variant_key, "FASE 6 post-proceso")
    _patch_config_fase6_constants(nb)
    _patch_evaluate_multiclass_cell(nb)
    _patch_multiclass_test_cell(nb)
    replace_cascade_checkpoint_paths(nb, cfg)
    patch_markdown_antes_de_ejecutar_output_dir(nb, cfg["output_dir"])
    prepend_interpretacion_warning(nb)
    append_execution_registry_cells(nb)
    clear_code_cell_outputs(nb)

    fname = f"train_spine_cascade_binary_to_thoracolumbar_v3_mejorafase6_postproceso_ligero_{variant_key}.ipynb"
    out = DST_DIR / fname
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for key in ("cpu", "cuda"):
        p = build_notebook(key)
        print("Escrito:", p)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Utilidades compartidas para generar notebooks de **mejoras** a partir de
`train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` (par `_cpu` / `_cuda`).

Convención de alcance (ver `README_MEJORAS_CASCADE.md` en esta carpeta):
- **Solo multiclase:** el parche toca únicamente código *después* de ROI / loaders
  y la celda de entrenamiento **multiclase** (p. ej. Fase 2 pesos CE, Fase 3 scheduler).
  Se puede armar un script pequeño que solo defina `VARIANTS` + un parche de texto.
- **Pre-multiclase o binario:** cambios en helpers, binario, splits o datos antes
  del multiclase (p. ej. Fase 1 letterbox en `prepare_multiclass_cascade_sample`).
  Suele requerir lógica adicional **no** cubierta solo por `patch_config_cell_*`.

Este módulo centraliza lo que es **común a todas las fases** actuales: mapa inicial,
Colab opcional, celda de configuración, rutas de checkpoints, markdown «Antes de
ejecutar», aviso de interpretación y limpieza de outputs.
"""
from __future__ import annotations

import copy
import uuid
from pathlib import Path
from typing import Any

COLAB_CUDA_PROJECT_ROOT = Path("/content/drive/Othercomputers/Mi portátil/ScoliosisSegmentation")

CUDA_CHECK_COLAB_CELL = """import torch

if torch.cuda.is_available():
    print("CUDA is available! GPU device:", torch.cuda.get_device_name(0))
    print("Number of CUDA devices:", torch.cuda.device_count())
else:
    print("CUDA is not available. Please ensure you have selected a GPU runtime.")
"""

DRIVE_MOUNT_COLAB_CELL = """try:
    from google.colab import drive

    drive.mount('/content/drive')
except ImportError:
    print(
        "[Info] No hay entorno Colab (google.colab): se omite drive.mount. "
        "En VS Code / Jupyter local el repo ya esta en disco; en Colab ejecuta esta celda para montar Drive."
    )
"""


def repo_root_prep_cell_source(*, include_colab_sync: bool = True) -> list[str]:
    """Celda 0: localiza la raiz del repo (contiene `data/Scoliosis_Dataset`) y hace os.chdir."""
    colab_block = ""
    if include_colab_sync:
        colab_block = f"""
    # Colab — Drive sincronizado (Other computers / Mi portatil)
    _colab_sync = Path(r"{COLAB_CUDA_PROJECT_ROOT.as_posix()}")
    if _colab_sync.is_dir() and (_colab_sync / marker).exists():
        return _colab_sync.resolve()
"""
    src = f'''import os
from pathlib import Path


def _resolve_project_root(marker: Path) -> Path:
    """Raiz del repo: carpeta que contiene `marker` (p. ej. data/Scoliosis_Dataset)."""
    env = os.environ.get("MAIA_PROJECT_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if (p / marker).exists():
            return p
    cwd = Path.cwd().resolve()
    for cand in [cwd, *cwd.parents]:
        if (cand / marker).exists():
            return cand
{colab_block}
    drive_nb = Path("/content/drive/MyDrive/Colab Notebooks")
    if drive_nb.is_dir():
        direct = drive_nb / "MAIA-PROYECTO"
        if (direct / marker).exists():
            return direct.resolve()
        for child in drive_nb.iterdir():
            try:
                if child.is_dir() and (child / marker).exists():
                    return child.resolve()
            except OSError:
                continue
    drive_maia = Path("/content/drive/MyDrive/MAIA-PROYECTO")
    if (drive_maia / marker).exists():
        return drive_maia.resolve()
    raise FileNotFoundError(
        f"No se encontro {{marker.as_posix()}}. Opciones: (1) montar Drive y %cd a la raiz del clon; "
        f"(2) os.environ['MAIA_PROJECT_ROOT'] = r'.../ScoliosisSegmentation'. cwd={{cwd}}"
    )


_MARKERS = [
    Path("data") / "Scoliosis_Dataset",
    Path("Scoliosis_Dataset_V3") / "Scoliosis_Dataset",
]
REPO = None
_last_err = None
for _m in _MARKERS:
    try:
        REPO = _resolve_project_root(_m)
        break
    except FileNotFoundError as e:
        _last_err = e
if REPO is None:
    raise FileNotFoundError(
        "No se encontro la raiz del repo (falta data/Scoliosis_Dataset o Scoliosis_Dataset_V3/...). "
        "En Colab: monta Drive, ejecuta la celda de montaje y %cd a ScoliosisSegmentation "
        "o define MAIA_PROJECT_ROOT."
    ) from _last_err

os.chdir(REPO)
print("Working directory (repo root):", Path.cwd())
'''
    return lines_from_str(src)


def lines_from_str(s: str) -> list[str]:
    parts = s.splitlines(keepends=True)
    return parts if parts else [""]


def new_code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines_from_str(source),
    }


def new_markdown_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": lines_from_str(text),
    }


def prepend_mapa_and_optional_colab_cuda_cells(nb: dict, variant_key: str, mapa_md: str) -> None:
    """Celda 0: intro. En `_cuda`, titulos markdown + CUDA + Drive + primera celda del base."""
    base_first = copy.deepcopy(nb["cells"][0])
    if variant_key == "cuda":
        nb["cells"][0]["source"] = lines_from_str(mapa_md)
        nb["cells"].insert(
            1,
            new_markdown_cell(
                "### [Colab — `_cuda`] Comprobar CUDA\n\n"
                "En **Google Colab** elige runtime **T4 GPU** (o superior). En VS Code / Jupyter local "
                "esta celda solo comprueba si `torch.cuda` ve una GPU.\n\n"
            ),
        )
        nb["cells"].insert(2, new_code_cell(CUDA_CHECK_COLAB_CELL))
        nb["cells"].insert(
            3,
            new_markdown_cell(
                "### [Colab — `_cuda`] Montar Google Drive\n\n"
                "Solo aplica en **Colab**. Fuera de Colab se captura `ImportError` y se imprime un mensaje; "
                "no hace falta montar Drive si el repo ya esta en disco.\n\n"
            ),
        )
        nb["cells"].insert(4, new_code_cell(DRIVE_MOUNT_COLAB_CELL))
        nb["cells"].insert(5, base_first)
    else:
        nb["cells"][0]["source"] = lines_from_str(mapa_md) + base_first["source"]


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


def find_config_cell_index(nb: dict) -> int:
    for j, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        t = "".join(c.get("source", []))
        if "OUTPUT_DIR = ROOT" in t and "MANIFEST_PATH" in t:
            return j
    raise RuntimeError("celda config no encontrada")


def patch_config_cell_for_training_variant(
    nb: dict,
    cfg: dict[str, str],
    variant_key: str,
    fase_tag_print: str,
) -> None:
    """
    Parchea la celda de configuración: OUTPUT_DIR (con `outputs/` o legado), resolución,
    batch, épocas, print de depuración e inyección Colab `_cuda`.
    """
    ic = find_config_cell_index(nb)
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

    _out_old_primary = "OUTPUT_DIR = ROOT / 'outputs' / 'analysis_outputs_v3' / 'training_runs_cascade_v3'"
    _out_old_legacy = "OUTPUT_DIR = ROOT / 'analysis_outputs_v3' / 'training_runs_cascade_v3'"
    _out_new = f"OUTPUT_DIR = ROOT / 'outputs' / 'analysis_outputs_v3' / '{cfg['output_dir']}'"
    if _out_old_primary in c4:
        c4 = c4.replace(_out_old_primary, _out_new, 1)
    elif _out_old_legacy in c4:
        c4 = c4.replace(
            _out_old_legacy,
            f"OUTPUT_DIR = ROOT / 'analysis_outputs_v3' / '{cfg['output_dir']}'",
            1,
        )
    else:
        raise RuntimeError("patron OUTPUT_DIR (training_runs_cascade_v3) no encontrado en celda config")

    c4 = c4.replace("IMG_SIZE_BINARY = (512, 256)", f"IMG_SIZE_BINARY = {cfg['img_binary']}")
    c4 = c4.replace("IMG_SIZE_MULTICLASS = (512, 256)", f"IMG_SIZE_MULTICLASS = {cfg['img_mc']}")
    c4 = c4.replace("BATCH_SIZE = 4", f"BATCH_SIZE = {cfg['batch']}")
    c4 = c4.replace("BINARY_EPOCHS = 14", f"BINARY_EPOCHS = {cfg['binary_epochs']}")
    c4 = c4.replace("MULTICLASS_EPOCHS = 24", f"MULTICLASS_EPOCHS = {cfg['multiclass_epochs']}")
    tag = f"{fase_tag_print} ({variant_key})"
    if tag not in c4:
        c4 = c4.replace(
            "print('ROOT:', ROOT)",
            f"print('{tag} — OUTPUT_DIR:', OUTPUT_DIR)\nprint('ROOT:', ROOT)",
        )
    if variant_key == "cuda":
        anchor = "    drive_nb = Path('/content/drive/MyDrive/Colab Notebooks')"
        if anchor in c4 and "Othercomputers" not in c4:
            block = (
                "    # [Colab _cuda] Drive sincronizado: Other computers / Mi portatil / ScoliosisSegmentation\n"
                f"    _colab_cuda_root = Path(r'{COLAB_CUDA_PROJECT_ROOT.as_posix()}')\n"
                "    if _colab_cuda_root.is_dir() and (_colab_cuda_root / marker).exists():\n"
                "        return _colab_cuda_root.resolve()\n"
            )
            c4 = c4.replace(anchor, block + anchor, 1)
    nb["cells"][ic]["source"] = lines_from_str(c4)


def replace_cascade_checkpoint_paths(nb: dict, cfg: dict[str, str]) -> None:
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


def patch_markdown_antes_de_ejecutar_output_dir(nb: dict, output_dir: str) -> None:
    for cell in nb["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        s = "".join(cell.get("source", []))
        if "## Antes de ejecutar" in s:
            s2 = s.replace(
                "`analysis_outputs_v3/training_runs_cascade_v3/`",
                f"`analysis_outputs_v3/{output_dir}/`",
            )
            cell["source"] = lines_from_str(s2)
            return


def clear_code_cell_outputs(nb: dict) -> None:
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


EXECUTION_REGISTRY_MARKDOWN = """## Registro de ejecución del notebook

La **fecha de última modificación** del archivo `.ipynb` en disco **no** indica cuándo terminó el entrenamiento: cambia en cada guardado.

**Recomendación:** al finalizar entrenamiento y evaluación, ejecutar la **celda de código siguiente** (la salida queda guardada en el notebook y marca el momento de esa ejecución). Opcionalmente completar la tabla.

| Campo | Valor |
|-------|--------|
| **Fecha de cierre del run** | *(ver salida de la celda siguiente o anotar)* |
| **Entorno** | *(p. ej. Colab T4, VS Code + CUDA, …)* |
"""

EXECUTION_REGISTRY_CODE = """# --- Registro de ejecución: ejecutar al finalizar entrenamiento y evaluación ---
from __future__ import annotations

from datetime import datetime, timezone

now_local = datetime.now().astimezone()
now_utc = datetime.now(timezone.utc)
print("Ejecución de esta celda — local:", now_local.strftime("%Y-%m-%d %H:%M:%S %Z"))
print("Ejecución de esta celda — UTC:  ", now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"))
"""


def append_execution_registry_cells(nb: dict) -> None:
    """Añade al final un bloque markdown + código para registrar cuándo se ejecutó el cierre del run."""
    for c in nb["cells"]:
        if c.get("cell_type") != "markdown":
            continue
        if "Registro de ejecución del notebook" in "".join(c.get("source", [])):
            return
    nb["cells"].append(new_markdown_cell(EXECUTION_REGISTRY_MARKDOWN))
    nb["cells"].append(new_code_cell(EXECUTION_REGISTRY_CODE))
    from mejoras_cierre_notebook_common import prepare_cierre_vacio_before_registry

    prepare_cierre_vacio_before_registry(nb)

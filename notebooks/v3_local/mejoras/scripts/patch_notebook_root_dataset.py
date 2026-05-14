# -*- coding: utf-8 -*-
import json
from pathlib import Path

ROOT_PROJ = Path(__file__).resolve().parents[2]

OLD = """ROOT = Path.cwd()
if not (ROOT / 'Scoliosis_Dataset_V3' / 'Scoliosis_Dataset').exists() and (ROOT.parent / 'Scoliosis_Dataset_V3' / 'Scoliosis_Dataset').exists():
    ROOT = ROOT.parent
"""

NEW = """import os


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


def to_lines(s: str) -> list[str]:
    p = s.splitlines(keepends=True)
    return p if p else [""]


def main() -> None:
    paths = [
        ROOT_PROJ / "train_spine_cascade_binary_to_thoracolumbar_v3.ipynb",
        ROOT_PROJ / "train_spine_cascade_binary_to_thoracolumbar_v3_inspection_ROI.ipynb",
        ROOT_PROJ / "train_spine_binary_and_thoracolumbar_v3.ipynb",
        ROOT_PROJ / "thoracolumbar_coverage_strategy_v3.ipynb",
    ]
    for p in paths:
        if not p.exists():
            print("missing", p)
            continue
        nb = json.loads(p.read_text(encoding="utf-8"))
        changed = False
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            s = "".join(cell["source"])
            if OLD in s:
                cell["source"] = to_lines(s.replace(OLD, NEW, 1))
                changed = True
        if changed:
            p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
            print("patched", p.name)
        else:
            print("skip (no OLD block)", p.name)


if __name__ == "__main__":
    main()

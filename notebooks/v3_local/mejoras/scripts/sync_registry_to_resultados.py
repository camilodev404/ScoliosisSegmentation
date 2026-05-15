# -*- coding: utf-8 -*-
"""Sincroniza la sección «Resumen final» de RESULTADOS desde experiment_registry.csv.

Fuente de verdad tabular: notebooks/v3_local/mejoras/experiment_registry.csv
Destino: notebooks/v3_local/mejoras/RESULTADOS_Y_DECISIONES_GENERAL.md

Solo incluye filas con columna `fecha` rellena (runs ejecutados).

Uso (desde la raíz del repo):
  python notebooks/v3_local/mejoras/scripts/sync_registry_to_resultados.py
"""
from __future__ import annotations

import csv
import re
from datetime import date
from pathlib import Path

MEJORAS = Path(__file__).resolve().parents[1]
REGISTRY = MEJORAS / "experiment_registry.csv"
RESULTADOS = MEJORAS / "RESULTADOS_Y_DECISIONES_GENERAL.md"

SECTION_START = "## Resumen final — registro de experimentos"

INTRO_STATIC = """\
Cuadro generado desde `notebooks/v3_local/mejoras/experiment_registry.csv` (**editar el CSV primero**, luego ejecutar este script).

**Alcance:** solo experimentos con **fecha de ejecución** (`fecha` no vacía en el CSV). La fila baseline (`0`) no se repite aquí; sus métricas test figuran arriba y en las columnas **Δ**. Variantes `_cpu` opcionales y pendientes siguen solo en el CSV. Métricas por clase (T9–T11, L5): [tabla §7.2](#tabla-comparativa-72--inventario-acumulado-actualizar-al-cerrar-cada-experimento) (Fase 0).

**Pipeline adoptado al cierre del plan (fases 1–8):** notebook vivo `notebooks/v3_local/train_spine_cascade_binary_to_thoracolumbar_v3.ipynb` con **Fase 3** (LR multiclase ×0,5 + `CosineAnnealingLR`) + **Fase 4** (augment ROI train). Referencia métrica: **`4_cuda`** — `macro_dice_fg` test **0,2380**. Checkpoints: `outputs/analysis_outputs_v3/training_runs_cascade_v3_fase4_augment_roi_cuda/`.
"""

FOOTER_STATIC = """\
**Columnas solo en el CSV:** `directorio_mejoras`, `notebook_o_script`, `output_metricas`, `notas`. Hipótesis completa en el CSV si esta tabla es demasiado ancha en el preview del editor.

### Dónde está el detalle de cada fase

| Necesidad | Dónde mirar |
|-----------|-------------|
| Decisión rápida y métricas globales | Este documento (§7.2 + tabla anterior) o el CSV |
| Análisis narrativo, tablas test, gráficas del run | Notebook `*_cuda.ipynb` de la fase + celdas «Análisis e interpretación» |
| Entradas/salidas y regenerar notebooks | `README.md` en cada carpeta `…_mejorafaseN_…/` |
| Convenciones, criterios de adopción, glosario informe | `PLAN_ACCION_AJUSTES_MODELOS.md` |
| Exportar a Excel / filtrar por fase o decisión | `experiment_registry.csv` |
"""


def _parse_float(val: str) -> float | None:
    val = (val or "").strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _fmt_num(val: str) -> str:
    f = _parse_float(val)
    if f is None:
        return (val or "").strip() or "—"
    return f"{f:.4f}".replace(".", ",")


def _fmt_delta(run_val: str, base_val: str) -> str:
    run_f = _parse_float(run_val)
    base_f = _parse_float(base_val)
    if run_f is None or base_f is None:
        return "—"
    d = run_f - base_f
    s = f"{d:+.4f}".replace(".", ",")
    return s


def _fmt_decision(decision: str, exp_id: str) -> str:
    d = (decision or "").strip()
    if not d:
        return "—"
    if d.lower() == "adoptar":
        return "**Adoptar**"
    if "no adoptar" in d.lower():
        return "**No adoptar**"
    if d.lower() in ("n/a baseline", "n/a"):
        return "N/A baseline"
    return d


def _format_hypothesis(s: str) -> str:
    """Texto completo (sin truncar) para la columna Hipótesis."""
    s = " ".join((s or "").split())
    if not s:
        return "—"
    return s.replace("|", "\\|")


def _sort_key(row: dict[str, str]) -> tuple:
    fase = row.get("fase", "")
    try:
        f = int(fase)
    except ValueError:
        f = 999
    return (f, row.get("id", ""))


def _bold_metrics_if_adopted(
    macro: str, iou: str, decision: str | None, exp_id: str
) -> tuple[str, str]:
    macro_f = _fmt_num(macro)
    iou_f = _fmt_num(iou)
    d = (decision or "").strip().lower()
    if d == "adoptar" or exp_id == "4_cuda":
        if macro_f != "—":
            macro_f = f"**{macro_f}**"
        if iou_f != "—":
            iou_f = f"**{iou_f}**"
    return macro_f, iou_f


def _executed_only(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if (r.get("fecha") or "").strip()]


def _baseline_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for r in rows:
        if (r.get("id") or "").strip() == "0":
            return r
    for r in rows:
        if (r.get("fase") or "").strip() == "0":
            return r
    return None


def _intro_with_baseline(baseline: dict[str, str] | None) -> str:
    intro = INTRO_STATIC
    if not baseline:
        return intro
    bd = _fmt_num(baseline.get("macro_dice_fg_test", ""))
    bi = _fmt_num(baseline.get("macro_iou_fg_test", ""))
    intro += (
        f"\n\n**Referencia baseline Fase 0 (`0`, cascada V3 core, test):** "
        f"`macro_dice_fg` **{bd}**, `macro_iou_fg` **{bi}**. "
        f"Columnas **Δ dice** / **Δ IoU**: diferencia del run frente a ese baseline."
    )
    return intro


def load_registry() -> list[dict[str, str]]:
    with REGISTRY.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if not (r.get("decision") or "").strip() and r.get("notas"):
            m = re.search(r";\s*No adoptar\s*$", r["notas"], re.I)
            if m:
                r["notas"] = r["notas"][: m.start()].rstrip()
                r["decision"] = "No adoptar"
    return rows


def build_section(rows: list[dict[str, str]]) -> str:
    executed = _executed_only(rows)
    executed = sorted(executed, key=_sort_key)
    baseline = _baseline_row(rows)
    base_dice = (baseline or {}).get("macro_dice_fg_test", "")
    base_iou = (baseline or {}).get("macro_iou_fg_test", "")
    lines = [
        SECTION_START,
        "",
        _intro_with_baseline(baseline),
        "",
        "| ID | Fase | Fecha | Sem. | `macro_dice_fg` | `macro_iou_fg` | Δ dice | Δ IoU | Decisión | Hipótesis |",
        "|----|------|-------|------|-----------------|----------------|--------|-------|----------|-----------|",
    ]
    for r in executed:
        exp_id = r.get("id", "")
        fecha = (r.get("fecha") or "").strip()
        seed = (r.get("seed") or "").strip() or "—"
        decision = _fmt_decision(r.get("decision", ""), exp_id)
        macro, iou = _bold_metrics_if_adopted(
            r.get("macro_dice_fg_test", ""),
            r.get("macro_iou_fg_test", ""),
            r.get("decision", ""),
            exp_id,
        )
        d_dice = _fmt_delta(r.get("macro_dice_fg_test", ""), base_dice)
        d_iou = _fmt_delta(r.get("macro_iou_fg_test", ""), base_iou)
        hip = _format_hypothesis(r.get("hipotesis", ""))
        lines.append(
            f"| `{exp_id}` | {r.get('fase', '')} | {fecha} | {seed} | {macro} | {iou} | {d_dice} | {d_iou} | {decision} | {hip} |"
        )
    lines.extend(["", FOOTER_STATIC, ""])
    lines.append(
        f"*Resumen final: {len(executed)} runs con fecha (de {len(rows)} filas en CSV). "
        f"Generado por `sync_registry_to_resultados.py` ({date.today().isoformat()}).*"
    )
    lines.append("")
    return "\n".join(lines)


def patch_resultados(section_md: str) -> bool:
    text = RESULTADOS.read_text(encoding="utf-8")
    if SECTION_START not in text:
        raise RuntimeError(f"No se encontró «{SECTION_START}» en {RESULTADOS.name}")
    pattern = re.compile(
        re.escape(SECTION_START) + r".*?(?=\n---\n\n\*Tabla §7\.2)",
        re.DOTALL,
    )
    new_text, n = pattern.subn(section_md.rstrip() + "\n", text, count=1)
    if n != 1:
        raise RuntimeError("No se pudo reemplazar la sección Resumen final (patrón no único)")
    RESULTADOS.write_text(new_text, encoding="utf-8")
    return True


def main() -> None:
    if not REGISTRY.is_file():
        raise FileNotFoundError(REGISTRY)
    rows = load_registry()
    section = build_section(rows)
    patch_resultados(section)
    n_exec = len(_executed_only(rows))
    print(f"OK: {n_exec} runs con fecha -> {RESULTADOS.name} (de {len(rows)} filas CSV)")


if __name__ == "__main__":
    main()

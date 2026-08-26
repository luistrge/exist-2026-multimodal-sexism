"""Sanitize presentation-only notebook residue without changing numeric evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
WINDOWS_USER = re.compile(r"C:\\Users\\[^\\\s]+")
WINDOWS_CACHE_COMMENT = re.compile(
    r"# Forzamos, no usamos setdefault: si el kernel hereda C:\\Users\\[^\n]+"
)
UNIX_EXPERIMENT_ROOT = re.compile(r"/home/[^/\s]+/LNR_CHG/EXIST2026_MEMES_ONLY_FINAL")
UNIX_WORKSPACE = re.compile(r"/home/[^/\s]+/LNR_CHG")
ENSEMBLE_NAME = "best_ensemble_development_model_selection"
ENSEMBLE_NAME_PATTERN = re.compile(
    r"\bbest_ensemble_(?:dev|development_model_selection(?:elopment_model_selection)*)\b"
)

CLEAR_OUTPUTS = {
    "01_eda_and_sensor_analysis.ipynb": {2},
    # Cells 13-14 contain multi-megabyte training/cache logs. The final ranking,
    # selected members, metrics, and confusion matrix remain in cells 15-17.
    "02_task21_binary_gate.ipynb": {2, 3, 4, 13, 14},
    "03_task22_source_intention.ipynb": {2},
    "04_task23_sexism_facets.ipynb": {2, 3, 4},
}

TEXT_REPLACEMENTS = {
    "/content/drive/MyDrive/LNR": "<CLOUD_WORKSPACE>",
    "D:\\LNR": "<LOCAL_WORKSPACE>",
    "Guardado:": "Saved:",
    "Cache de scores reutilizada para": "Reused score cache for",
    "Distribución de idiomas train": "Train language distribution",
    "Distribución de idiomas test": "Test language distribution",
    "Imágenes train encontradas": "Train images found",
    "Imágenes test encontradas": "Test images found",
    "Textos vacíos train": "Empty training OCR texts",
    "Número de features sensoriales": "Number of sensor features",
    "Origen idioma train:": "Training language source:",
    "Origen idioma test:": "Test language source:",
    "Task2.2 oficial:": "Official Task 2.2:",
    "N columnas visuales:": "Visual feature count:",
    "ET · tiempo reacción": "ET · reaction time",
    "no hay columnas baseline/prev": "no baseline/previous columns",
    "Candidatos totales": "Total candidates",
    "Candidatos robustos": "Stable candidates",
    "Pool robusto": "Stable pool",
    "Stable pool: 31 candidatos": "Stable pool: 31 candidates",
    "Total de ensambles evaluados": "Total evaluated ensembles",
    "Ranking completo": "Complete ranking",
    "Complete ranking (ordenado por calib_macro_f1):": (
        "Complete ranking (sorted by calibration macro-F1):"
    ),
    "[Stage A] Texto clasico": "[Stage A] Classical text",
    "[Stage B] Texto + sensores": "[Stage B] Text + sensors",
    "[Stage C] Sensores solos": "[Stage C] Sensor-only",
    "[Stage D] Imagen sola": "[Stage D] Image-only",
    "[Stage E] Texto embedding": "[Stage E] Text embeddings",
    "[Stage F] Texto + imagen embedding": "[Stage F] Text + image embeddings",
    "[Stage G] Multimodal + sensores": "[Stage G] Multimodal + sensors",
    "(todos los candidatos individuales)": "(all individual candidates)",
    "(todos los ensambles probados)": "(all evaluated ensembles)",
    "Filas Task 2.3 v2 con al menos una faceta: 1990 de 3984": (
        "Task 2.3 v2 rows with at least one facet: 1990 of 3984"
    ),
    "Artefactos facet-only generados. No se ejecuta el validador oficial completo porque "
    "falta la clase NO; se combinan despues con Task 2.1.": (
        "Facet-only artifacts generated. The complete organizer validator is not run "
        "because the NO class is absent; combine them with Task 2.1 first."
    ),
    "Stacking diagnóstico entrenado en dev; no usar como métrica oficial:": (
        "Diagnostic stacking trained on development data; not an official metric:"
    ),
    "Validador oficial localizado:": "Organizer validator located:",
    "El script oficial valida por carpetas y nombres de run; deja los JSON generados "
    "en OUTPUT_DIR para revisión/envío.": (
        "The organizer script validates run folders and names; generated JSON files remain "
        "in OUTPUT_DIR for review/submission."
    ),
    "Terminado": "Finished",
    "tiempo=": "time=",
}

SOURCE_REPLACEMENTS = {
    "674-example validation split": "674-example development/model-selection split",
    "357 sexist-only development examples": "357 sexist-only development/model-selection examples",
    "398 facet-positive development examples": (
        "398 facet-positive development/model-selection examples"
    ),
    "At the validation threshold of **0.371642**": (
        "On the development/model-selection set, at the threshold of **0.371642**"
    ),
    "The selected `geom_mean_top5` ensemble reaches **0.637 conditional macro-F1**": (
        "The selected `geom_mean_top5` ensemble reaches **0.637 conditional "
        "development/model-selection macro-F1**"
    ),
    "The selected eight-member soft vote uses label-specific thresholds and reaches "
    "**0.677 facet macro-F1**": (
        "The selected eight-member soft vote uses label-specific thresholds and reaches "
        "**0.677 development/model-selection facet macro-F1**"
    ),
    '"model": "best_ensemble_dev"': '"model": "best_ensemble_development_model_selection"',
    '    path = GOLD_ROOT / f"EXIST2025_training_task{task_id}_gold_{kind}.json"\n'
    "    if not path.exists():\n"
    "        return {}": (
        '    matches = sorted(GOLD_ROOT.glob(f"*training_task{task_id}_gold_{kind}.json"))\n'
        "    if len(matches) != 1:\n"
        "        return {}\n"
        "    path = matches[0]"
    ),
    '_DEFAULT_WORKSPACE_ROOT = Path("D:/LNR") if Path("D:/LNR").exists() else Path.cwd()': (
        "_DEFAULT_REPO_ROOT = Path.cwd().resolve()\n"
        'if _DEFAULT_REPO_ROOT.name == "notebooks":\n'
        "    _DEFAULT_REPO_ROOT = _DEFAULT_REPO_ROOT.parent\n"
        "_DEFAULT_WORKSPACE_ROOT = _DEFAULT_REPO_ROOT.parent"
    ),
}

LEGACY_SOURCE_LINE_MARKERS = (
    "# En Colab esta carpeta debe existir exactamente",
    "# Rutas de datos. En este equipo:",
    'WORKSPACE_ROOT / "EXIST2026_memes"',
    'WORKSPACE_ROOT / "EXIST2026_final_project"',
    'PROJECT_ROOT / "EXIST2026_memes"',
    'PROJECT_ROOT / "EXIST2026_final_project"',
    'Path("/content/drive/MyDrive/LNR',
)

TASK23_WARNING = (
    "\n> **Evaluation scope.** The same 398-example development set ranks experts, "
    "selects the top eight, tunes per-label thresholds, and produces 0.677. This is "
    "development/model-selection performance, not an independent holdout estimate.\n"
)


def sanitize_text(value: str) -> str:
    value = ANSI.sub("", value)
    value = WINDOWS_USER.sub("<USER_HOME>", value)
    value = UNIX_EXPERIMENT_ROOT.sub("<EXPERIMENT_ROOT>", value)
    value = UNIX_WORKSPACE.sub("<LOCAL_WORKSPACE>", value)
    for old, new in TEXT_REPLACEMENTS.items():
        value = value.replace(old, new)
    return ENSEMBLE_NAME_PATTERN.sub(ENSEMBLE_NAME, value)


def sanitize_output_value(value: Any, key: str | None = None) -> Any:
    if key and (key.startswith("image/") or key in {"application/pdf"}):
        return value
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, list):
        return [sanitize_output_value(item, key) for item in value]
    if isinstance(value, dict):
        return {name: sanitize_output_value(item, name) for name, item in value.items()}
    return value


def curate(notebook: dict[str, Any], name: str) -> dict[str, Any]:
    for index, cell in enumerate(notebook["cells"]):
        if index in CLEAR_OUTPUTS.get(name, set()) and cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

        source = "".join(cell.get("source", []))
        for old, new in SOURCE_REPLACEMENTS.items():
            source = source.replace(old, new)
        source = WINDOWS_CACHE_COMMENT.sub(
            "# Override inherited cache paths so every artifact stays under OUTPUT_DIR.", source
        )
        for old, new in TEXT_REPLACEMENTS.items():
            source = source.replace(old, new)
        source = "".join(
            line
            for line in source.splitlines(keepends=True)
            if not any(marker in line for marker in LEGACY_SOURCE_LINE_MARKERS)
        )
        source = ENSEMBLE_NAME_PATTERN.sub(ENSEMBLE_NAME, source)
        if name == "04_task23_sexism_facets.ipynb" and index == 0 and TASK23_WARNING not in source:
            source += TASK23_WARNING
        cell["source"] = source.splitlines(keepends=True)

        if cell.get("outputs"):
            cell["outputs"] = sanitize_output_value(cell["outputs"])
            for output in cell["outputs"]:
                if "text" in output:
                    output["text"] = [sanitize_text(line) for line in output["text"]]
                plain = output.get("data", {}).get("text/plain")
                if plain:
                    output["data"]["text/plain"] = [sanitize_text(line) for line in plain]
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if curation would change files.")
    args = parser.parse_args()
    changed: list[str] = []
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        original = json.loads(path.read_text(encoding="utf-8"))
        curated = curate(original, path.name)
        serialized = json.dumps(curated, ensure_ascii=False, indent=1) + "\n"
        if serialized != path.read_text(encoding="utf-8"):
            changed.append(path.name)
            if not args.check:
                path.write_text(serialized, encoding="utf-8")
    if args.check and changed:
        raise SystemExit(f"Notebook curation required: {', '.join(changed)}")
    print(
        "Notebook curation "
        + ("check passed." if args.check else f"updated {len(changed)} file(s).")
    )


if __name__ == "__main__":
    main()

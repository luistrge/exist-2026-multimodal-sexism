"""Lightweight structural and report-alignment checks for the public repository."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_SHA256 = "344ce1d849258436ac7db65d7f9c3d94c4ca979286fb2715db50eb8a896a8c00"

REQUIRED_FILES = [
    "README.md",
    "CITATION.cff",
    "requirements.txt",
    "requirements-heavy.txt",
    "docs/METHODOLOGY.md",
    "docs/REPORT_ALIGNMENT.md",
    "docs/REPRODUCIBILITY.md",
    "docs/MODEL_CARD.md",
    "report/serrano-team-exist2026-report.pdf",
    "results/validation_summary.csv",
    "results/sensor_ablation.csv",
    "results/task23_per_facet.csv",
    "results/submission_distribution.csv",
    "scripts/exist2026_meme_utils.py",
    "scripts/task2_2_metric_focused_pipeline.py",
]

NOTEBOOKS = [
    "00_project_overview.ipynb",
    "01_eda_and_sensor_analysis.ipynb",
    "02_task21_binary_gate.ipynb",
    "03_task22_source_intention.ipynb",
    "04_task23_sexism_facets.ipynb",
]

EXPECTED_METRICS = {
    ("task2_1", "macro_f1"): 0.709,
    ("task2_1", "f1_yes"): 0.815,
    ("task2_1", "recall_yes"): 0.940,
    ("task2_1", "accuracy"): 0.748,
    ("task2_2", "conditional_macro_f1"): 0.637,
    ("task2_2", "f1_judgemental"): 0.469,
    ("task2_2", "macro_f1"): 0.485,
    ("task2_3", "facet_macro_f1"): {0.677, 0.674},
    ("task2_3", "micro_f1"): 0.700,
    ("task2_3", "samples_f1"): 0.694,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_files() -> None:
    for relative in REQUIRED_FILES:
        require((ROOT / relative).is_file(), f"Missing required file: {relative}")
    for notebook in NOTEBOOKS:
        require((ROOT / "notebooks" / notebook).is_file(), f"Missing notebook: {notebook}")
    require(not list(ROOT.rglob("*.zip")), "Submission archives must not be committed")


def validate_report() -> None:
    report = ROOT / "report" / "serrano-team-exist2026-report.pdf"
    digest = hashlib.sha256(report.read_bytes()).hexdigest()
    require(digest == REPORT_SHA256, "The final report PDF differs from the audited source")


def validate_notebooks() -> None:
    parsed = {}
    for name in NOTEBOOKS:
        path = ROOT / "notebooks" / name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        require(notebook.get("nbformat") == 4, f"Unsupported notebook format: {name}")
        require(isinstance(notebook.get("cells"), list) and notebook["cells"], f"Empty notebook: {name}")
        require(notebook["cells"][0].get("cell_type") == "markdown", f"Missing narrative header: {name}")
        parsed[name] = notebook

    # Outputs from superseded intermediate runs are deliberately absent.
    task22 = parsed["03_task22_source_intention.ipynb"]
    task23 = parsed["04_task23_sexism_facets.ipynb"]
    require(task22["cells"][23].get("outputs") == [], "Partial VLM cache output must remain excluded")
    require(task23["cells"][20].get("outputs") == [], "Superseded Task 2.3 distribution must remain excluded")

    overview = "".join(cell_text(cell) for cell in parsed["00_project_overview.ipynb"]["cells"])
    for token in ("0.709", "0.637", "0.485", "0.677", "0.674"):
        require(token in overview, f"Overview is missing report value {token}")


def cell_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def validate_metrics() -> None:
    with (ROOT / "results" / "validation_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        values.setdefault((row["task"], row["metric"]), []).append(float(row["value"]))
    for key, expected in EXPECTED_METRICS.items():
        actual = values.get(key, [])
        require(actual, f"Missing report metric: {key}")
        if isinstance(expected, set):
            require(set(actual) == expected, f"Unexpected values for {key}: {actual}")
        else:
            require(expected in actual, f"Unexpected value for {key}: {actual}")

    with (ROOT / "results" / "submission_distribution.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        distribution = {
            (row["task"], row["label"]): int(row["count"])
            for row in csv.DictReader(handle)
        }
    require(distribution[("task2_1", "YES")] == 608, "Task 2.1 YES count differs from report")
    require(distribution[("task2_1", "NO")] == 445, "Task 2.1 NO count differs from report")
    require(distribution[("task2_2", "DIRECT")] == 396, "Task 2.2 DIRECT count differs from report")
    require(distribution[("task2_2", "JUDGEMENTAL")] == 212, "Task 2.2 JUDGEMENTAL count differs from report")
    require(distribution[("task2_3", "STEREOTYPING-DOMINANCE")] == 420, "Task 2.3 count differs from report")


def validate_readme() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for relative in (
        "report/serrano-team-exist2026-report.pdf",
        "notebooks/00_project_overview.ipynb",
        "assets/figures/final_pipeline_overview.png",
    ):
        require(relative in readme, f"README does not link to {relative}")
        require((ROOT / relative).exists(), f"Broken local README target: {relative}")
    require("Top-5" in readme and "LNR" in readme, "Top-5 LNR recognition is missing")
    require("hidden test-set scores" in readme, "Evaluation-scope disclaimer is missing")


def main() -> None:
    validate_files()
    validate_report()
    validate_notebooks()
    validate_metrics()
    validate_readme()
    print("Repository validation passed.")


if __name__ == "__main__":
    main()

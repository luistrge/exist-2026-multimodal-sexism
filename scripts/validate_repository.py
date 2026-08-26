"""Lightweight structural and report-alignment checks for the public repository."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from audit_notebook_report_alignment import main as audit_notebook_report_alignment

ROOT = Path(__file__).resolve().parents[1]
REPORT_SHA256 = "344ce1d849258436ac7db65d7f9c3d94c4ca979286fb2715db50eb8a896a8c00"

REQUIRED_FILES = [
    "README.md",
    "CITATION.cff",
    "pyproject.toml",
    "requirements.txt",
    "requirements-heavy.txt",
    ".github/workflows/quality.yml",
    "docs/METHODOLOGY.md",
    "docs/EVALUATION_PROTOCOL.md",
    "docs/EVIDENCE_AUDIT.md",
    "docs/LICENSING.md",
    "docs/REPORT_ALIGNMENT.md",
    "docs/REPRODUCIBILITY.md",
    "docs/MODEL_CARD.md",
    "report/serrano-team-exist2026-report.pdf",
    "results/validation_summary.csv",
    "results/sensor_ablation.csv",
    "results/task23_per_facet.csv",
    "results/submission_distribution.csv",
    "results/report_traceability.csv",
    "results/evaluation_scope.csv",
    "results/official_leaderboard.csv",
    "results/RESULTS.md",
    "results/reproducible_baseline_task21.json",
    "scripts/audit_notebook_report_alignment.py",
    "scripts/curate_notebook_outputs.py",
    "scripts/exist2026_meme_utils.py",
    "scripts/task2_2_metric_focused_pipeline.py",
    "src/exist2026/__init__.py",
    "src/exist2026/cli.py",
    "src/exist2026/config.py",
    "src/exist2026/data.py",
    "src/exist2026/evaluation.py",
    "src/exist2026/features.py",
    "src/exist2026/model.py",
    "src/exist2026/training.py",
    "tests/conftest.py",
    "tests/test_data.py",
    "tests/test_evaluation.py",
    "tests/test_features.py",
    "tests/test_workflow.py",
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

EXPECTED_SENSOR_ABLATIONS = {
    ("task2_1", "E5+CLIP LightGBM"): (0.678, 0.706, 0.028),
    ("task2_2", "sparse word-character logistic regression"): (0.538, 0.508, -0.030),
    ("task2_2", "dense MPNet+CLIP"): (0.528, 0.552, 0.024),
    ("task2_3", "sparse facet branch"): (0.605, 0.580, -0.025),
}

EXPECTED_FACETS = {
    "IDEOLOGICAL-INEQUALITY": (168, 0.71, 0.89, 0.79),
    "OBJECTIFICATION": (187, 0.69, 0.89, 0.78),
    "STEREOTYPING-DOMINANCE": (200, 0.58, 0.90, 0.70),
    "SEXUAL-VIOLENCE": (96, 0.60, 0.75, 0.67),
    "MISOGYNY-NON-SEXUAL-VIOLENCE": (78, 0.33, 0.71, 0.45),
}

EXPECTED_SUBMISSION_DISTRIBUTION = {
    ("task2_1", "YES"): 608,
    ("task2_1", "NO"): 445,
    ("task2_2", "NO"): 445,
    ("task2_2", "DIRECT"): 396,
    ("task2_2", "JUDGEMENTAL"): 212,
    ("task2_3", "NO"): 445,
    ("task2_3", "STEREOTYPING-DOMINANCE"): 420,
    ("task2_3", "IDEOLOGICAL-INEQUALITY"): 293,
    ("task2_3", "OBJECTIFICATION"): 273,
    ("task2_3", "MISOGYNY-NON-SEXUAL-VIOLENCE"): 270,
    ("task2_3", "SEXUAL-VIOLENCE"): 194,
}

EXPECTED_SCOPED_METRICS = {
    ("task2_1", "eight_member_soft_vote", "macro_f1"): 0.709011,
    ("task2_1", "eight_member_soft_vote", "icm"): 0.148542,
    ("task2_1", "eight_member_soft_vote", "icm_soft"): -0.044835,
    ("task2_2", "geom_mean_top5", "macro_f1"): 0.636674,
    ("task2_3", "top8_soft_vote", "facet_macro_f1"): 0.677110,
    ("task2_3", "top8_soft_vote", "icm"): 0.003900,
    ("task2_3", "top8_soft_vote", "icm_soft"): -5.945134,
    ("task2_1", "reproducible_tfidf_visual_baseline", "macro_f1"): 0.660175,
}

EXPECTED_BASELINE_METRICS = {
    "macro_f1": 0.6601752064032268,
    "f1_yes": 0.710594315245478,
    "precision_yes": 0.7372654155495979,
    "recall_yes": 0.685785536159601,
    "accuracy": 0.6676557863501483,
}

EXPECTED_OFFICIAL_RESULTS = {
    ("task2_1", "soft-soft"): (
        "SerranoTeam_1",
        14,
        139,
        -0.0801,
        0.4871,
        "cross_entropy",
        0.8969,
        6,
    ),
    ("task2_1", "hard-hard"): ("SerranoTeam_1", 21, 212, 0.2053, 0.6044, "f1_yes", 0.7389, 7),
    ("task2_2", "soft-soft"): (
        "SerranoTeam_1",
        10,
        112,
        -1.1095,
        0.3820,
        "cross_entropy",
        1.4263,
        8,
    ),
    ("task2_2", "hard-hard"): ("SerranoTeam_3", 26, 181, -0.2319, 0.4194, "macro_f1", 0.4430, 9),
    ("task2_3", "soft-soft"): (
        "SerranoTeam_1",
        11,
        113,
        -4.8477,
        0.2431,
        "cross_entropy",
        2.1461,
        10,
    ),
    ("task2_3", "hard-hard"): ("SerranoTeam_2", 16, 182, -0.6375, 0.3677, "macro_f1", 0.4712, 11),
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
        require(
            isinstance(notebook.get("cells"), list) and notebook["cells"], f"Empty notebook: {name}"
        )
        require(
            notebook["cells"][0].get("cell_type") == "markdown", f"Missing narrative header: {name}"
        )
        parsed[name] = notebook

    # Outputs from superseded intermediate runs are deliberately absent.
    task21 = parsed["02_task21_binary_gate.ipynb"]
    task22 = parsed["03_task22_source_intention.ipynb"]
    task23 = parsed["04_task23_sexism_facets.ipynb"]
    require(
        task22["cells"][23].get("outputs") == [], "Partial VLM cache output must remain excluded"
    )
    require(
        task22["cells"][43].get("outputs") == [], "Auxiliary Task 2.1 headline must remain excluded"
    )
    require(
        task22["cells"][45].get("outputs") == [],
        "Superseded Task 2.2 distribution must remain excluded",
    )
    require(
        task21["cells"][13].get("outputs") == [],
        "Task 2.1 dense training logs must remain excluded",
    )
    require(
        task21["cells"][14].get("outputs") == [], "Task 2.1 transformer logs must remain excluded"
    )
    require(
        task23["cells"][20].get("outputs") == [],
        "Superseded Task 2.3 distribution must remain excluded",
    )
    require("expected_task21" in cell_text(task21["cells"][15]), "Task 2.1 report guard is missing")
    require("expected_task22" in cell_text(task22["cells"][45]), "Task 2.2 report guard is missing")
    require(
        "This is not the selected Task 2.1 submission system." in cell_text(task22["cells"][43]),
        "Task 2.2 auxiliary gate must be scoped explicitly",
    )
    require("expected_task23" in cell_text(task23["cells"][20]), "Task 2.3 report guard is missing")
    require(
        "EXIST2026_EXPORT_FINAL_SUBMISSIONS" in cell_text(task23["cells"][20]),
        "Task 2.3 final export must remain opt-in",
    )

    overview = "".join(cell_text(cell) for cell in parsed["00_project_overview.ipynb"]["cells"])
    for token in ("0.709", "0.637", "0.485", "0.677", "0.674"):
        require(token in overview, f"Overview is missing report value {token}")
    for token in ("#14/139", "#10/112", "#11/113"):
        require(token in overview, f"Overview is missing official position {token}")
    require("Top-5" not in overview, "Superseded Top-5 claim remains in overview")

    full_text = "\n".join(nested_strings(parsed))
    for residue in (
        "D:/LNR",
        "D:\\LNR",
        "/home/",
        "/content/drive/MyDrive/LNR",
        "C:\\Users\\",
        "\x1b[",
    ):
        require(residue not in full_text, f"Notebook residue remains: {residue}")

    require(
        "development/model-selection" in cell_text(task21["cells"][0]),
        "Task 2.1 headline scope is missing",
    )
    require(
        "development/model-selection" in cell_text(task22["cells"][0]),
        "Task 2.2 headline scope is missing",
    )
    task23_header = cell_text(task23["cells"][0])
    require("development/model-selection" in task23_header, "Task 2.3 headline scope is missing")
    require(
        "same 398-example development set" in task23_header
        and "not an independent holdout estimate" in task23_header,
        "Task 2.3 model-selection warning is missing",
    )


def cell_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def nested_strings(value: object) -> list[str]:
    """Flatten notebook JSON strings so residue checks see decoded paths."""

    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in nested_strings(item)]
    if isinstance(value, dict):
        return [text for item in value.values() for text in nested_strings(item)]
    return []


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
            (row["task"], row["label"]): int(row["count"]) for row in csv.DictReader(handle)
        }
    require(
        distribution == EXPECTED_SUBMISSION_DISTRIBUTION,
        f"Submission distribution differs from report: {distribution}",
    )

    with (ROOT / "results" / "sensor_ablation.csv").open(newline="", encoding="utf-8") as handle:
        sensor_rows = {
            (row["task"], row["branch"]): (
                float(row["without_sensors"]),
                float(row["with_sensors"]),
                float(row["delta"]),
            )
            for row in csv.DictReader(handle)
        }
    require(
        sensor_rows == EXPECTED_SENSOR_ABLATIONS,
        f"Sensor ablations differ from report: {sensor_rows}",
    )

    with (ROOT / "results" / "task23_per_facet.csv").open(newline="", encoding="utf-8") as handle:
        facet_rows = {
            row["facet"]: (
                int(row["support"]),
                float(row["precision"]),
                float(row["recall"]),
                float(row["f1"]),
            )
            for row in csv.DictReader(handle)
        }
    require(facet_rows == EXPECTED_FACETS, f"Per-facet results differ from report: {facet_rows}")

    with (ROOT / "results" / "evaluation_scope.csv").open(newline="", encoding="utf-8") as handle:
        scope_rows = list(csv.DictReader(handle))
    scoped = {(row["task"], row["system"], row["metric"]): row for row in scope_rows}
    for key, expected in EXPECTED_SCOPED_METRICS.items():
        require(key in scoped, f"Missing scoped metric: {key}")
        actual = float(scoped[key]["value"])
        require(
            math.isclose(actual, expected, abs_tol=1e-9),
            f"Unexpected scoped value for {key}: {actual}",
        )
    for row in scope_rows:
        status = row["official_leaderboard_status"]
        if status == "official leaderboard":
            require(
                row["data_scope"].startswith("official hidden test")
                and row["metric_implementation"].startswith("official EXIST 2026 overview"),
                "An official metric is missing organizer provenance",
            )
        else:
            require(
                status in {"not available in repository", "not submitted"},
                f"Unknown leaderboard status: {status}",
            )
    task23_scope = scoped[("task2_3", "top8_soft_vote", "facet_macro_f1")]
    require(
        task23_scope["selection_role"] == "expert ranking and per-label threshold model selection",
        "Task 2.3 selection reuse is not recorded",
    )

    with (ROOT / "results" / "official_leaderboard.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        official_rows = {
            (row["task"], row["protocol"]): (
                row["run"],
                int(row["rank"]),
                int(row["participant_runs"]),
                float(row["icm"]),
                float(row["icm_norm"]),
                row["secondary_metric"],
                float(row["secondary_value"]),
                int(row["source_table"]),
            )
            for row in csv.DictReader(handle)
        }
    require(
        official_rows == EXPECTED_OFFICIAL_RESULTS,
        f"Official leaderboard values changed: {official_rows}",
    )


def validate_baseline() -> None:
    path = ROOT / "results" / "reproducible_baseline_task21.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    protocol = result["protocol"]
    require(protocol["seed"] == 42, "Baseline seed changed")
    require(protocol["train_examples"] == 2696, "Baseline training split changed")
    require(protocol["validation_examples"] == 674, "Baseline validation split changed")
    require(protocol["threshold"] == 0.5, "Baseline fixed threshold changed")
    require(protocol["threshold_tuned_on_validation"] is False, "Baseline threshold leaked")
    for metric, expected in EXPECTED_BASELINE_METRICS.items():
        actual = float(result["metrics"][metric])
        require(
            math.isclose(actual, expected, abs_tol=1e-12), f"Baseline {metric} changed: {actual}"
        )
    require(
        result["confusion_matrix"]["values"] == [[175, 98], [126, 275]],
        "Baseline confusion matrix changed",
    )
    require(
        "not a submitted system" in result["relationship_to_report"],
        "Baseline/report boundary is missing",
    )


def validate_readme() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for relative in (
        "report/serrano-team-exist2026-report.pdf",
        "notebooks/00_project_overview.ipynb",
        "assets/figures/final_pipeline_overview.png",
    ):
        require(relative in readme, f"README does not link to {relative}")
        require((ROOT / relative).exists(), f"Broken local README target: {relative}")
    require("Top-5" not in readme, "Superseded Top-5 claim remains in README")
    for position in ("#14 of 139", "#10 of 112", "#11 of 113"):
        require(position in readme, f"README is missing official position {position}")
    require("hidden test-set scores" in readme, "Evaluation-scope disclaimer is missing")
    require("20-second TL;DR" in readme, "README TL;DR is missing")
    require("Best official soft run" in readme, "Official/internal metric separation is missing")
    require("0.660 held-out macro-F1" in readme, "Reproducible baseline result is missing")
    require("No open-source license has been applied" in readme, "Licensing status is missing")
    for heading in (
        "## Experimental reasoning and decision logic",
        "## Task 2.1 — Protecting downstream recall",
        "## Task 2.2 — Separating endorsement from criticism",
        "## Task 2.3 — Modelling facets as a multilabel problem",
        "## Evidence, traceability, and credibility",
    ):
        require(heading in readme, f"README is missing section: {heading}")
    require(len(readme.split()) >= 2500, "README must retain the extended technical narrative")


def main() -> None:
    validate_files()
    validate_report()
    validate_notebooks()
    validate_metrics()
    validate_baseline()
    validate_readme()
    audit_notebook_report_alignment()
    print("Repository validation passed.")


if __name__ == "__main__":
    main()

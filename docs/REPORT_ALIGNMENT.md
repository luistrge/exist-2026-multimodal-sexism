# Report alignment

The final project PDF is canonical for internal experiment claims. The official
EXIST 2026 overview is canonical for hidden-test rankings. This document maps both
evidence sources to the curated repository.

| Report topic | Primary notebook | Repository summary |
|---|---|---|
| Dataset, OCR, image, soft-label, and physiological analysis | `01_eda_and_sensor_analysis.ipynb` | `00_project_overview.ipynb`, `README.md` |
| Task 2.1 eight-expert gate, threshold, confusion matrix, and sensor ablation | `02_task21_binary_gate.ipynb` | `results/validation_summary.csv`, `results/sensor_ablation.csv` |
| Task 2.2 candidate audit, stable pool, selected geometric ensemble, and routing loss | `03_task22_source_intention.ipynb` | `results/validation_summary.csv`, `docs/METHODOLOGY.md` |
| Task 2.3 top-eight facet ensemble, per-facet metrics, and positive-only routing audit | `04_task23_sexism_facets.ipynb` | `results/task23_per_facet.csv`, `results/validation_summary.csv` |
| Final test prediction counts | Final PDF only | `results/submission_distribution.csv` |
| Metric scope and internal PyEvALL values | Executed Task 2.1/2.3 outputs | `results/evaluation_scope.csv`, `docs/EVALUATION_PROTOCOL.md` |
| Official soft-soft and hard-hard results | EXIST 2026 overview Tables 6–11 | `results/RESULTS.md`, `results/official_leaderboard.csv` |
| Reproducible engineering baseline | Independent CPU rerun | `src/exist2026/`, `results/reproducible_baseline_task21.json` |

The field-level mapping is available in `results/report_traceability.csv`, and the
notebook claims can be checked automatically with:

```bash
python scripts/audit_notebook_report_alignment.py
```

## Curation decisions

- The four latest executed analysis/modelling notebooks were retained and renamed in reading order.
- Earlier unexecuted templates were excluded.
- Intermediate submission artifacts and distributions superseded by the report were excluded.
- A superseded Task 2.2 cascade distribution was replaced with a report-aligned submission audit because it came from an intermediate gate rather than the primary submitted run.
- A stale Task 2.3 cascade output was removed from the curated notebook because it did not represent the final reported submission.
- Final-export cells now compare their hard distributions with the canonical report counts and reject mismatched intermediate runs.
- The report PDF was copied verbatim and is not regenerated from older LaTeX sources.
- English narrative cells were added while preserving the audited code and valid execution evidence.

## Numerical policy

Internal experiment tables reproduce the final report; official ranking tables
reproduce the organizer's overview. Intermediate hyperparameter comparisons remain
inside the notebooks, but no superseded run is presented as the final system.

The repository claims hidden-test performance only for the explicitly archived
official runs. Report headline F1 metrics remain labelled development/model-selection
performance, routing figures remain diagnostics, and PyEvALL by itself identifies
only a metric implementation.

See `docs/EVIDENCE_AUDIT.md` for the distinction between executed output,
executable source, and report-only submission evidence.

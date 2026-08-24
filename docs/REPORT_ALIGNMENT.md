# Report alignment

The final PDF is the canonical source for all public headline values. This document maps each report block to the curated evidence retained in the repository.

| Report topic | Primary notebook | Repository summary |
|---|---|---|
| Dataset, OCR, image, soft-label, and physiological analysis | `01_eda_and_sensor_analysis.ipynb` | `00_project_overview.ipynb`, `README.md` |
| Task 2.1 eight-expert gate, threshold, confusion matrix, and sensor ablation | `02_task21_binary_gate.ipynb` | `results/validation_summary.csv`, `results/sensor_ablation.csv` |
| Task 2.2 candidate audit, stable pool, selected geometric ensemble, and routing loss | `03_task22_source_intention.ipynb` | `results/validation_summary.csv`, `docs/METHODOLOGY.md` |
| Task 2.3 top-eight facet ensemble, per-facet metrics, and positive-only routing audit | `04_task23_sexism_facets.ipynb` | `results/task23_per_facet.csv`, `results/validation_summary.csv` |
| Final test prediction counts | Final PDF only | `results/submission_distribution.csv` |

## Curation decisions

- The four latest executed analysis/modelling notebooks were retained and renamed in reading order.
- Earlier unexecuted templates were excluded.
- Intermediate submission artifacts and distributions superseded by the report were excluded.
- A stale Task 2.3 cascade output was removed from the curated notebook because it did not represent the final reported submission.
- The report PDF was copied verbatim and is not regenerated from older LaTeX sources.
- English narrative cells were added while preserving the audited code and valid execution evidence.

## Numerical policy

The public tables in `README.md`, `docs/`, and `results/` reproduce the final report. Intermediate hyperparameter comparisons remain inside the notebooks as experimental evidence, but no superseded run is presented as the final system.

The repository does not claim hidden test-set scores. Reported metrics are internal validation or held-out routing diagnostics.

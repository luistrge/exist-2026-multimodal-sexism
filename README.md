# Multimodal Sexism Identification and Characterization in Memes

**Serrano Team · Top-5 project in the LNR EXIST 2026 challenge**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)](notebooks/00_project_overview.ipynb)
[![Tasks](https://img.shields.io/badge/EXIST_2026-Tasks_2.1_·_2.2_·_2.3-6C4CCF)](https://nlp.uned.es/exist2026/)

This repository presents Serrano Team's hierarchical multimodal system for the three [EXIST 2026](https://nlp.uned.es/exist2026/) meme tasks: binary sexism identification, source-intention classification, and multilabel sexism-facet classification. The project combines OCR text, meme images, multilingual language models, and physiological signals from eye tracking, heart rate, and EEG.

The work was recognized as a **Top-5 project in the LNR challenge** built around EXIST 2026. The complete technical account is available in the [final report](report/serrano-team-exist2026-report.pdf). All headline values in this repository are aligned with that report.

![Hierarchical multimodal pipeline](assets/figures/final_pipeline_overview.png)

## Results at a glance

| Component | Evaluation scope | Selected result |
|---|---|---:|
| Task 2.1 gate | 674-example validation split | **0.709 macro-F1** |
| Task 2.1 gate | `YES` class | **0.815 F1**, **0.94 recall** |
| Task 2.1 gate | Overall | **0.748 accuracy** |
| Task 2.2 ensemble | 357 sexist-only development examples | **0.637 macro-F1** |
| Task 2.2 ensemble | `JUDGEMENTAL` class | **0.469 F1** |
| Task 2.2 cascade | Three-class routing diagnostic | **0.485 macro-F1** |
| Task 2.3 ensemble | 398 facet-positive development examples | **0.677 macro-F1** |
| Task 2.3 ensemble | Multilabel | **0.700 micro-F1**, **0.694 samples-F1** |
| Task 2.3 + gate | Positive-only routing diagnostic | **0.674 macro-F1** |

These are internal validation and held-out diagnostic results on labelled training data, not hidden test-set scores. Conditional and routed values answer different questions and are kept separate throughout the project.

## System design

The final system is intentionally hierarchical:

1. **Task 2.1 — Binary gate.** An eight-expert soft vote predicts `NO` or `YES`. The threshold is tuned for high `YES` recall so downstream labels are not lost prematurely.
2. **Task 2.2 — Source intention.** A five-member geometric-mean ensemble predicts `DIRECT` or `JUDGEMENTAL` only after a positive gate decision.
3. **Task 2.3 — Sexism facets.** An eight-expert soft vote predicts one or more of five facets using label-specific thresholds, again only after a positive gate decision.

Late fusion proved more reliable than a single monolithic architecture. OCR supplies the most stable signal; image embeddings provide meme context; language-specific transformers help with idioms and offensive phrasing; physiological descriptors are retained only when an ablation supports them.

## Repository guide

| Path | Purpose |
|---|---|
| [`notebooks/00_project_overview.ipynb`](notebooks/00_project_overview.ipynb) | Recommended entry point and report-aligned summary |
| [`notebooks/01_eda_and_sensor_analysis.ipynb`](notebooks/01_eda_and_sensor_analysis.ipynb) | Data, OCR, image, uncertainty, and sensor audit |
| [`notebooks/02_task21_binary_gate.ipynb`](notebooks/02_task21_binary_gate.ipynb) | Multimodal `NO/YES` gate and eight-expert ensemble |
| [`notebooks/03_task22_source_intention.ipynb`](notebooks/03_task22_source_intention.ipynb) | Candidate benchmark, robustness audit, and selected ensemble |
| [`notebooks/04_task23_sexism_facets.ipynb`](notebooks/04_task23_sexism_facets.ipynb) | Multilabel facet models and routing diagnostic |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Compact technical explanation of the final system |
| [`docs/REPORT_ALIGNMENT.md`](docs/REPORT_ALIGNMENT.md) | Traceability from report claims to notebooks and tables |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Environment, data layout, caches, and execution modes |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Intended use, limitations, and responsible-use notes |
| [`results/`](results) | Report-aligned metrics, ablations, and submission summaries |

The curated sequence contains the latest executed notebooks that support the final report. Earlier templates, stale submissions, and intermediate numerical summaries are intentionally excluded.

## Dataset

The audited release contains **3,984 training memes** and **1,053 test memes**. The training set includes **2,005 English** and **1,979 Spanish** examples, with OCR text, images, and physiological measurements.

The dataset is not redistributed here. Obtain it through the official EXIST organizers and place it beside this repository as described in [the reproducibility guide](docs/REPRODUCIBILITY.md).

## Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Start with `notebooks/00_project_overview.ipynb`, then continue in numerical order. Install `requirements-heavy.txt` only when reproducing transformer, dense-embedding, or GPU branches.

## Key findings

- A recall-oriented Task 2.1 gate protects downstream coverage: only 26 of 401 sexist validation memes are routed to `NO`.
- Physiological features help selectively. They improve one Task 2.1 LightGBM branch by **+0.028 macro-F1**, but reduce the tested sparse Task 2.2 and Task 2.3 branches.
- Task 2.2 is the main routing bottleneck: 114 of 357 sexist development examples are lost before intention classification in the diagnostic cascade.
- Minority facets remain difficult. `MISOGYNY-NON-SEXUAL-VIOLENCE` reaches **0.45 F1**, well below the more frequent ideological-inequality and objectification facets.

![Task 2.1 gate diagnostics](assets/figures/task21_gate_diagnostics.png)

![Task 2.3 per-facet behaviour](assets/figures/task23_facet_bubble.png)

## Responsible use

This is a research system, not a moderation service. The data contain sexist and potentially harmful content; predictions can reproduce annotation bias and can vary across language, culture, and meme context. Any operational use requires independent validation, human review, subgroup analysis, and a clear appeal process.

## Citation

If this repository supports your work, cite the technical report:

```text
Serrano Team. (2026). Multimodal Sexism Identification and
Characterization in Memes: Technical Report for EXIST 2026
Tasks 2.1, 2.2 and 2.3.
```

The “Top-5” designation refers to the LNR challenge ranking associated with this project.

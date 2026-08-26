# Reproducibility

## What is reproducible from this repository

The notebooks preserve the audited experiment structure and execution evidence. Classical text, handcrafted image, sensor aggregation, thresholding, metrics, and ensemble logic can be rerun from the official dataset. Heavy transformer and embedding branches require model downloads or external caches that are intentionally not committed.

The repository separates two goals:

- **Read and audit:** open the notebooks as checked in; executed outputs document the experiment selected for the report.
- **Run the public baseline:** train and evaluate the CPU-only OCR/visual Logistic Regression system without external model caches.
- **Recompute:** install the required dependencies, obtain the official data, and provide the relevant caches or allow model downloads.

## Public baseline

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -c constraints-baseline.txt -e .

exist2026-baseline train \
  --data-root "/absolute/path/to/EXIST 2026 Dataset V0.2" \
  --output-dir outputs/baseline-task21

exist2026-baseline evaluate \
  --data-root "/absolute/path/to/EXIST 2026 Dataset V0.2" \
  --model-dir outputs/baseline-task21
```

The model uses a fixed threshold of 0.5; it does not tune that threshold on the
674-example holdout. The audited seed-42 run is recorded in
`results/reproducible_baseline_task21.json`. `constraints-baseline.txt` pins the
runtime dependency versions used to generate that artifact; `pyproject.toml`
retains supported ranges for normal installation.

## Expected data layout

Place the official release beside the repository:

```text
LNR/
├── EXIST 2026 Dataset V0.2/
│   ├── EXIST 2026 Memes Dataset/
│   │   ├── training/
│   │   │   ├── EXIST2026_training.json
│   │   │   └── memes/
│   │   └── test/
│   │       ├── EXIST2026_test_clean.json
│   │       └── memes/
│   └── evaluation/
│       ├── golds/
│       └── exist2025_format_val_V0.2.py
└── exist-2026-multimodal-sexism/
```

The utility module discovers this layout automatically. Alternative locations can be configured with:

```bash
export EXIST2026_DATASET_ROOT="/absolute/path/to/EXIST 2026 Dataset V0.2"
export EXIST2026_MEMES_ROOT="/absolute/path/to/EXIST 2026 Memes Dataset"
export EXIST2026_EVAL_ROOT="/absolute/path/to/evaluation"
```

The release supplied for the 2026 challenge retains an `EXIST2025` prefix in its
validator, gold filenames, and required submission `test_case` field. This is an
upstream compatibility constraint, not a stale experiment reference. Repository
loaders discover gold files by task suffix and isolate the submission value behind
the `ORGANIZER_LEGACY_TEST_CASE` constant.

The standalone Task 2.1 and Task 2.3 notebooks also accept `LNR_WORKSPACE`, `LNR_PROJECT_ROOT`, and task-specific output-directory variables documented in their setup cells.

## Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[notebooks]"
```

For dense embeddings, transformers, tree boosters, and GPU branches:

```bash
pip install -e ".[notebooks,heavy]"
```

Open Jupyter from the repository root so relative paths resolve consistently:

```bash
jupyter lab
```

## Notebook order

1. `00_project_overview.ipynb`
2. `01_eda_and_sensor_analysis.ipynb`
3. `02_task21_binary_gate.ipynb`
4. `03_task22_source_intention.ipynb`
5. `04_task23_sexism_facets.ipynb`

Task 2.2 depends on `scripts/exist2026_meme_utils.py` and `scripts/task2_2_metric_focused_pipeline.py`. The latter restores the interface used by the audited master notebook.

## External caches

The full audited run used artifacts such as frozen image/text embeddings, fine-tuned transformer probability arrays, VLM descriptions, and model checkpoints. They are omitted because of size, licensing, and portability. The notebooks skip unavailable optional branches where possible; an exact full rerun requires caches with IDs aligned to the official data release.

Typical cache families include:

- multilingual E5 and MPNet text embeddings;
- CLIP, DINOv2, and ViT image embeddings;
- cached fine-tuned transformer probabilities;
- `task2_2_vlm_reasoning.csv`;
- Task 2.1 checkpoints used by downstream gate diagnostics.

Do not compare a partial rerun with the reported ensemble unless the same retained expert set is available.

## Validation checks

Run the lightweight repository audit before committing changes:

```bash
python scripts/audit_notebook_report_alignment.py
python scripts/validate_repository.py
```

The notebook/report audit checks selected members, preserved metrics, confusion
rows, routing losses, and final-distribution guards directly in the notebook JSON.
The repository validator checks notebook validity, required files, report-aligned
CSV values, excluded stale artifacts, README evidence sections, and the byte
identity of the final PDF.

Run the software-quality suite with:

```bash
pip install -e ".[dev]"
ruff check src tests
black --check src tests
pytest --cov=exist2026 --cov-report=term-missing
```

The tests use synthetic data and are safe to run in CI without the official
release. GitHub Actions runs them on Python 3.11 and 3.12.

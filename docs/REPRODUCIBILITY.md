# Reproducibility

## What is reproducible from this repository

The notebooks preserve the audited experiment structure and execution evidence. Classical text, handcrafted image, sensor aggregation, thresholding, metrics, and ensemble logic can be rerun from the official dataset. Heavy transformer and embedding branches require model downloads or external caches that are intentionally not committed.

The repository separates two goals:

- **Read and audit:** open the notebooks as checked in; executed outputs document the experiment selected for the report.
- **Recompute:** install the required dependencies, obtain the official data, and provide the relevant caches or allow model downloads.

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

The standalone Task 2.1 and Task 2.3 notebooks also accept `LNR_WORKSPACE`, `LNR_PROJECT_ROOT`, and task-specific output-directory variables documented in their setup cells.

## Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For dense embeddings, transformers, tree boosters, and GPU branches:

```bash
pip install -r requirements-heavy.txt
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
python scripts/validate_repository.py
```

This checks notebook validity, required files, report-aligned CSV values, excluded stale artifacts, and the presence of the final PDF.

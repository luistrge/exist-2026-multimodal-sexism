# Evidence and notebook audit

## Audit objective

The final technical report is the canonical source for public claims. The notebooks are accepted as supporting evidence only when their code or preserved outputs reproduce the model composition, validation scope, or rounded value described in that report.

The audit distinguishes three evidence types:

1. **Executed output:** metrics, model rankings, confusion matrices, or classifications preserved in a notebook execution.
2. **Executable source:** the code that constructs the reported split, model family, ensemble, threshold, or export guard.
3. **Report-aligned narrative/table:** a value that belongs to the final submission audit but cannot be reconstructed without the exact external model caches or submitted JSON files.
4. **Organizer-issued result:** a hidden-test rank or metric published in the official EXIST 2026 overview, tied to a run and evaluation protocol.

This distinction matters. A final submission count is not a performance metric, and a conditional downstream score is not equivalent to a routed cascade score.

A fourth distinction is now explicit: PyEvALL ICM/ICM-Soft calculated on internal
labels are not organizer-issued leaderboard results. `results/evaluation_scope.csv`
records both the data role and metric implementation.

The official overview tables were checked separately. Serrano Team's best soft
positions are #14, #10, and #11 for Tasks 2.1–2.3; they are preserved in
`results/official_leaderboard.csv` and explained in `results/RESULTS.md`.

## What was checked

- Dataset sizes and language counts.
- Task-specific split sizes and evaluation scopes.
- Exact retained-member lists for all three tasks.
- Task 2.1 threshold, headline metrics, and confusion matrix.
- Task 2.2 candidate/stability counts, selected ensemble, minority-class F1, and routing loss.
- Task 2.3 headline metrics and every per-facet precision/recall/F1/support row.
- Cross-task physiological ablations.
- Final hard-run distributions from the report.
- Byte identity of the published PDF against the audited source.
- Official soft-soft and hard-hard run positions and metrics from overview Tables 6–11.
- Whether every headline is labelled as development/model selection, routing
  diagnostic, fixed-threshold holdout baseline, or official leaderboard result.

The internal report mapping is in `results/report_traceability.csv`; official
leaderboard provenance is in `results/official_leaderboard.csv`. Run:

```bash
python scripts/audit_notebook_report_alignment.py
```

The script checks the notebook source and text outputs directly and fails if required evidence disappears.

## Superseded output removed during the second audit

The original Task 2.2 master notebook ended with a diagnostic cascade export driven by an intermediate Task 2.1 gate. That cell produced a test distribution different from the primary submitted run documented in the final report. It was useful during experimentation but was not valid final-submission evidence.

The curated notebook now replaces that executed block with an explicit audit against `results/submission_distribution.csv`. Task 2.1 and Task 2.3 exports also contain report-distribution guards: an export that does not match the final report is rejected as an intermediate run.

The same Task 2.2 section used a leakage-controlled auxiliary gate to quantify
routing loss. Its standalone validation output was removed because that auxiliary
model is not the selected eight-member Task 2.1 system and could be mistaken for
the Task 2.1 headline result. The executable calculation remains, its scope is now
explicit, and the report-supported downstream evidence—114 lost examples and
0.485 routed macro-F1—remains preserved.

No intermediate distribution is presented as a final result.

## Reproducibility boundary

The lightweight checks are fully reproducible from the repository. Exact regeneration of every dense expert requires the original embedding arrays, VLM-description cache, transformer probability arrays, and checkpoints. Those artifacts are external because of their size and model/data licensing constraints.

The repository therefore makes a precise claim: it preserves the audited experiment logic and executed evidence, validates every public report value, and documents which heavy artifacts are required for a complete retraining run. It does not claim that a cache-free laptop run will reconstruct the full ensemble.

# Evaluation protocol and claim boundaries

## Why this document exists

The final technical report is an accurate record of the selected 2026 experiments,
but its headline values are not blind leaderboard scores. The repository therefore
separates model-selection evidence, internal PyEvALL metrics, routing diagnostics,
and organizer-issued leaderboard results.

## Metric scopes

| Scope | Meaning | Permitted claim |
|---|---|---|
| Development/model-selection | The partition influenced expert selection, aggregation, thresholding, or another reported choice | Evidence used to select the reported system |
| Fixed-threshold held-out baseline | The partition was not used to choose baseline features, model, or threshold | Reproducible internal holdout estimate for that fixed baseline |
| Routing diagnostic | A targeted check of error propagation, sometimes on a constructed or positive-only subset | Diagnostic result only; not full cascade performance |
| Internal PyEvALL | ICM/ICM-Soft computed with the organizer's metric implementation on labelled internal data | Internal ICM/ICM-Soft, never official leaderboard performance |
| Official leaderboard | Score returned by the challenge organizers on hidden test labels | Official only when the original organizer result is archived |

The machine-readable mapping is in `results/evaluation_scope.csv`.

## Archived official results

The official EXIST 2026 overview supplies hidden-test rankings for every submitted
run. Serrano Team's best soft-soft positions are #14/139 for Task 2.1, #10/112 for
Task 2.2, and #11/113 for Task 2.3. The corresponding hard-hard best positions are
#21/212, #26/181, and #16/182. These are run-level, all-language rankings.

`results/RESULTS.md` records the complete values and interpretation;
`results/official_leaderboard.csv` is the machine-readable source table.

## Task 2.3 limitation

The reported Task 2.3 experiment trains experts on 1,592 examples and produces
scores for a 398-example development set. That same development set influences:

1. expert ranking and selection of the top eight;
2. one threshold per facet; and
3. the reported 0.677 facet macro-F1.

The value is therefore **development/model-selection performance**. It is retained
because it is the value documented in the final report, but it is not an independent
estimate of generalization.

## Protocol for a new independent Task 2.3 estimate

A corrected rerun should freeze the trained expert pool and partition examples into
three roles using multilabel and language stratification:

1. **Fit:** train each expert.
2. **Calibration/model selection:** rank experts, select the ensemble, and tune the
   five thresholds.
3. **Evaluation:** apply the frozen ensemble and thresholds exactly once.

If retraining every heavy expert is impractical, the 398 cached development-score
rows can be partitioned into calibration and evaluation subsets. Expert ranking and
threshold tuning must then see only calibration rows. This produces a cleaner
estimate than the reported protocol, although the smaller evaluation set will have
higher variance.

Any new number must be published beside—not silently substituted for—the
report-aligned 0.677 value.

## Official metric implementation versus official result

PyEvALL is used inside the notebooks to calculate ICM and ICM-Soft. Calling the
implementation “official” does not make a score an official challenge result. The
repository uses the names `internal PyEvALL ICM` and `internal PyEvALL ICM-Soft`.
Official values are labelled with their submitted run, protocol, hidden-test scope,
and overview-paper table. This provenance—not use of the PyEvALL package—is what
makes them official.

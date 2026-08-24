# Methodology

## Problem formulation

EXIST 2026 defines three linked meme tasks. Task 2.1 detects sexism (`NO`/`YES`); Task 2.2 assigns the source intention (`DIRECT`/`JUDGEMENTAL`) to sexist memes; Task 2.3 assigns one or more sexism facets. Because the downstream labels are valid only for sexist items, the final system mirrors this hierarchy.

All values below come from the final technical report. They are internal validation or held-out diagnostic values on labelled training data.

## Task 2.1 — Binary gate

The gate uses a stratified 2,696/674 train/development split. Its selected system is an equal-weight soft vote over eight experts:

1. Multilingual E5 + CLIP, logistic regression (`C=1`).
2. Multilingual E5 + CLIP, XGBoost.
3. Cardiff offensive-language RoBERTa.
4. Multilingual E5 + CLIP + sensors, LightGBM.
5. Multilingual E5 + DINOv2, logistic regression (`C=3`).
6. Multilingual E5 + CLIP, LightGBM.
7. Multilingual E5 + DINOv2, logistic regression (`C=1`).
8. Multilingual E5 + CLIP, logistic regression (`C=3`).

The `YES` threshold is 0.371642. The selected gate reaches 0.709 macro-F1, 0.815 F1 for `YES`, 0.94 `YES` recall, and 0.748 accuracy. Its confusion matrix is:

| Gold \ Prediction | `NO` | `YES` |
|---|---:|---:|
| `NO` | 129 | 144 |
| `YES` | 26 | 375 |

The low false-negative count is deliberate: a false `NO` prevents both downstream tasks from recovering.

## Task 2.2 — Source intention

Task 2.2 uses only the `DIRECT` and `JUDGEMENTAL` examples, partitioned into 1,111 fit, 314 calibration, and 357 development items. The benchmark evaluates 37 candidates. A candidate enters the stable pool when development macro-F1 is at least 0.50 and the absolute calibration–development gap is no larger than 0.08; 31 candidates survive.

The selected `geom_mean_top5` ensemble combines:

1. VLM-enriched word/character TF-IDF + logistic regression.
2. XLM-R + rich Qwen description, soft-label training.
3. VLM-enriched word/character TF-IDF + ComplementNB.
4. Multilingual DistilBERT, soft-label training.
5. OCR word/character TF-IDF + ComplementNB.

The ensemble reaches 0.637 conditional macro-F1 and 0.469 F1 for `JUDGEMENTAL`. In the three-class routing diagnostic, macro-F1 falls to 0.485 because 114 of 357 sexist development examples are stopped by the binary gate.

## Task 2.3 — Sexism facets

Task 2.3 is trained on 1,990 sexist memes with at least one facet, using a 1,592/398 train/development split. The selected top-eight soft vote uses label-specific thresholds and combines:

1. E5 + CLIP MLP.
2. Cardiff RoBERTa for English.
3. E5 + CLIP one-vs-rest logistic regression (`C=1`).
4. BETO for Spanish.
5. E5 text one-vs-rest logistic regression (`C=3`).
6. E5 + CLIP one-vs-rest logistic regression (`C=3`).
7. E5 + CLIP classifier-chain logistic regression (`C=1`).
8. E5 text one-vs-rest logistic regression (`C=1`).

The selected ensemble reaches 0.677 facet macro-F1, 0.700 micro-F1, and 0.694 samples-F1. A positive-only gate audit yields 0.674 macro-F1; this is a coverage diagnostic, not a complete mixed `NO`/facet evaluation.

## Physiological-signal ablation

Physiological features are treated as optional expert inputs, not universally beneficial features.

| Branch | Without sensors | With sensors | Change |
|---|---:|---:|---:|
| Task 2.1 E5 + CLIP LightGBM | 0.678 | 0.706 | **+0.028** |
| Task 2.2 sparse word/character LR | 0.538 | 0.508 | **−0.030** |
| Task 2.2 dense MPNet + CLIP | 0.528 | 0.552 | **+0.024** |
| Task 2.3 sparse facet branch | 0.605 | 0.580 | **−0.025** |

The final systems therefore retain one sensor-aware Task 2.1 expert, use sensor branches as Task 2.2 candidates rather than a mandatory final input, and keep the Task 2.3 ensemble sensor-free.

## Evaluation scope

Conditional scores assume that an item has already reached a downstream classifier. Routed scores include gate errors. The two scopes must not be treated as interchangeable. The report avoids describing these development values as cross-validation or hidden-test performance.

## Main limitations

- Routing errors propagate and cannot be repaired downstream.
- VLM descriptions depend on visual reasoning quality and cache consistency.
- Rare facets, especially non-sexual-violence misogyny, remain the primary bottleneck.
- Internal validation does not establish out-of-domain or operational reliability.

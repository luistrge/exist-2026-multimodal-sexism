# Model card

## Model overview

The project is a research pipeline for multilingual sexism analysis in memes. It uses a binary gate followed by conditional intention and multilabel facet ensembles. Inputs may include OCR text, image embeddings, language-specific transformer outputs, and aggregated physiological signals.

## Intended use

- Research on multimodal sexism identification and characterization.
- Reproduction and comparison within the EXIST 2026 experimental setting.
- Study of hierarchical routing, expert ensembles, and selective sensor fusion.

## Out-of-scope use

- Fully automated moderation or punitive decisions.
- Profiling individuals from physiological measurements.
- Deployment outside the documented English/Spanish meme setting without new validation.
- Treating model output as a factual judgement about a person or community.

## Training and evaluation data

The official training release contains 3,984 memes: 2,005 in English and 1,979 in Spanish. The test release contains 1,053 memes. Data and annotations are not redistributed in this repository.

The report-aligned F1 scores are development/model-selection values or routing
diagnostics on labelled training data. They are separate from the official
organizer-issued hidden-test results. The Task 2.3 development set is used for both
expert/threshold selection and the reported internal 0.677 calculation.

## Performance summary

- Official soft-soft run ranks: #14/139 (Task 2.1), #10/112 (Task 2.2), and #11/113 (Task 2.3).
- Official hard-hard best run ranks: #21/212, #26/181, and #16/182, respectively.
- Task 2.1 development/model selection: 0.709 macro-F1; 0.815 F1 and 0.94 recall for `YES`.
- Task 2.2 development/model selection: 0.637 conditional macro-F1; 0.485 routed diagnostic macro-F1.
- Task 2.3 development/model selection: 0.677 facet macro-F1; 0.700 micro-F1; 0.694 samples-F1.
- Reproducible Task 2.1 CPU baseline: 0.660 macro-F1 on a fixed-threshold holdout.

Official ranks are run-level and all-language. Full ICM values and provenance are
in `results/RESULTS.md`; internal and official metrics must not be compared as if
they used the same labels or selection protocol.

## Limitations and risks

- Errors at the binary gate propagate to both downstream tasks.
- Meme meaning can depend on culture, context, irony, and text-image interaction not captured by the models.
- The rarest facets have substantially weaker F1.
- Annotator disagreement and social bias can be learned by the system.
- Physiological features introduce privacy and consent concerns and should never be repurposed for individual profiling.
- Performance may differ across language, dialect, demographic references, image style, and distribution shift.
- Development/model-selection reuse may make the report-aligned headline values
  optimistic, especially Task 2.3.

## Recommended safeguards

Use human review, document uncertainty, monitor subgroup behaviour, keep a correction/appeal mechanism, and run an independent data-protection and ethical review before any operational use.

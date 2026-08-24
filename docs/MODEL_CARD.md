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

All reported scores are internal validation or held-out routing diagnostics on labelled training data. They are not hidden test-set scores.

## Performance summary

- Task 2.1: 0.709 macro-F1; 0.815 F1 and 0.94 recall for `YES`.
- Task 2.2: 0.637 conditional macro-F1; 0.485 routed diagnostic macro-F1.
- Task 2.3: 0.677 facet macro-F1; 0.700 micro-F1; 0.694 samples-F1.

## Limitations and risks

- Errors at the binary gate propagate to both downstream tasks.
- Meme meaning can depend on culture, context, irony, and text-image interaction not captured by the models.
- The rarest facets have substantially weaker F1.
- Annotator disagreement and social bias can be learned by the system.
- Physiological features introduce privacy and consent concerns and should never be repurposed for individual profiling.
- Performance may differ across language, dialect, demographic references, image style, and distribution shift.

## Recommended safeguards

Use human review, document uncertainty, monitor subgroup behaviour, keep a correction/appeal mechanism, and run an independent data-protection and ethical review before any operational use.

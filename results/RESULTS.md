# Official EXIST 2026 results

Serrano Team submitted runs to all three EXIST 2026 meme subtasks. The team's
best official position in each task came from the soft-soft evaluation:

| Task | Best run | Official position | ICM-Soft | ICM-Soft Norm | Cross entropy |
|---|---|---:|---:|---:|---:|
| Task 2.1 — Sexism identification | `SerranoTeam_1` | **#14 of 139 participant runs** | -0.0801 | 0.4871 | 0.8969 |
| Task 2.2 — Source intention | `SerranoTeam_1` | **#10 of 112 participant runs** | -1.1095 | 0.3820 | 1.4263 |
| Task 2.3 — Sexism categorization | `SerranoTeam_1` | **#11 of 113 participant runs** | -4.8477 | 0.2431 | 2.1461 |

These positions are official all-language run rankings, not unique-team rankings.
The denominator counts participant submissions and excludes the organizer's gold
reference and two non-informative baselines.

## Best hard-hard runs

For completeness, these are Serrano Team's best discrete-label runs in the same
official overview:

| Task | Best run | Official position | ICM-Hard | ICM-Hard Norm | Reported F1 |
|---|---|---:|---:|---:|---:|
| Task 2.1 | `SerranoTeam_1` | #21 of 212 participant runs | 0.2053 | 0.6044 | 0.7389 `F1(YES)` |
| Task 2.2 | `SerranoTeam_3` | #26 of 181 participant runs | -0.2319 | 0.4194 | 0.4430 macro-F1 |
| Task 2.3 | `SerranoTeam_2` | #16 of 182 participant runs | -0.6375 | 0.3677 | 0.4712 macro-F1 |

## Source and interpretation

The source is the official [EXIST 2026 overview paper](https://clef-staging.pages.dev/paper152.pdf),
Tables 6–11. `results/official_leaderboard.csv` preserves the values in a
machine-readable form.

The official metrics above were calculated by the organizers on hidden test labels.
They must not be mixed with the report's 0.709, 0.637, and 0.677 internal
development/model-selection F1 values. Internal PyEvALL output only identifies the
metric implementation; it is not an official result. The precise boundary between
these evidence classes is documented in `docs/EVALUATION_PROTOCOL.md`.

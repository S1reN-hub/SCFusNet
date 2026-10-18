# Reproducibility statement for submission

## Code and data availability

Code, derived case-level inputs, split manifests, model artifacts, per-record predictions, and verification reports for SCFusNet are publicly available through the versioned repository archive at https://doi.org/10.5281/zenodo.23446670 and the linked GitHub releases. The Zenodo version preserves the complete current code/results package and recovered model supplement; the four bulk-input archives are hosted on the fixed GitHub release and identified by archived sizes and SHA-256 checksums. Upstream data access conditions and attribution requirements continue to apply.

The original monolithic training checkpoint could not be retrieved. Expert parameters were recovered from a preserved epoch-zero frozen checkpoint, whose protocol records the original expert-checkpoint digest and matched input-normalization digest. The released expert checkpoint is a new serialization, accompanied by its unchanged extraction source, freeze-policy evidence, checksums, exact matched input normalization, the preserved formal dynamic MLP heads, and separate AOD correction checkpoints. The unavailable original built-in router and conditional modules are not reconstructed or used by the released external-head inference route.

Raw-input inference was verified on all 119 test cases for fusion seeds 42, 43 and 44. The original 601-record main-comparison cohort, 452-record smart-persistence intersection, and 38-hour AOD-admitted subset were retained. Recomputed main comparison metrics agree with the manuscript at the reported precision; unrounded differences are documented. The AOD admission set is unchanged, and its admitted-hour RMSE reduction remains 11.40% after rounding. This verification involved no retraining or model selection. It establishes numerical compatibility of the released inference route, rather than bit-exact reconstruction of the original execution or complete training reproducibility.

## Scope notes for editors and reviewers

The repository supplies a single command for the NPZ-to-experts-to-fusion-to-AOD workflow and original-cohort scoring, an exact package-version record, per-seed outputs, and provenance/integrity reports. The main comparison uses forecasts before the separate AOD correction. The AOD sensitivity analysis remains exploratory because its test split had previously been inspected.

The cause of the unavailable original checkpoint is not asserted: expiration of a remote instance has been suggested but has not been independently established. No newly trained checkpoint is represented as the original model. Claims of original file-byte identity, complete historical executed-source/environment identity, or full retraining reproducibility are not made.

For manuscript use, retain the factual limitations and cite the version actually used. Journal-specific availability fields may require the Code and Data Availability text to be split into separate statements.

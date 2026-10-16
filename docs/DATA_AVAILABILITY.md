# Data availability

This release accompanies the 2026-10-16 manuscript version of SCFusNet and provides derived research artifacts found in the local project at packaging time (2026-10-16).

## Included material

- Case-level split manifests for 1,462 station–cycle cases: 1,224 training, 119 validation, and 119 testing cases.
- Frozen branch feature caches and targets, fusion-head checkpoints for seeds 42/43/44, and exported predictions.
- AOD-enriched feature caches, candidate-specific AOD correction weights, admission diagnostics, and all nine sensitivity candidates.
- Matched daytime prediction tables, smart-persistence comparison material, summary metrics, data-quality audits, and ablation result exports.
- Historical launch and partial recovered remote snapshots, which differ and are not an exact complete formal-training snapshot.

The public artifacts support independent recalculation of released metrics and verified inference checks for the cached fusion heads. AOD feature caches, correction weights, and historical scripts are included for inspection. The artifacts do not supply all original observation and forecast fields.

## Material absent from this snapshot

The complete raw 1,462-case multimodal dataset and the trained satellite/NWP backbone checkpoint were not located in the available local project. Consequently, independent reproduction of raw-input backbone inference, backbone training, and the complete end-to-end experiment is not supported by this release alone. No unavailable file is represented as downloadable here.

The full manuscript, correspondence, account credentials, and unrelated project material are outside the release. Manuscript version identification uses bibliographic metadata and a source-file hash.

## Upstream provenance

| Component | Source or access point | Role |
| --- | --- | --- |
| Ground GHI | [BSRN](https://bsrn.awi.de/data/), [SURFRAD](https://gml.noaa.gov/grad/stardata.html) | Hourly supervision and evaluation labels |
| Satellite | GOES, Himawari, [GridSat-B1](https://www.ncei.noaa.gov/products/gridded-geostationary-brightness-temperature) | Recent cloud morphology and motion; source varies by station coverage |
| NWP | GFS-driven WRF-Chem simulations, 9-km and 3-km domains | Multilayer atmospheric fields and surface diagnostics |
| Aerosol | [CAMS Atmosphere Data Store](https://ads.atmosphere.copernicus.eu/) | AOD at 550 nm |
| Solar geometry | Derived from station location and target time | Daylight, clear-sky reference, and air mass |

Use the original providers' access procedures and attribution terms when obtaining source data. BSRN source files retain their BSRN-1.0 notices and dataset DOI information where present. The release does not claim ownership of upstream observations or forecasts, and no blanket data license is imposed on them.

## Interpretation of released subsets

The 601-record common daytime comparison, the 452-record smart-persistence intersection, and the 38-hour AOD-admitted subset answer different evaluation questions. Counts are unique target-hour records, not multiplied by the three model seeds. AOD admission spans 10 unique test cases. Split manifests contain potential geometrically eligible hours before all label and persistence checks; these counts need not equal final metric cohort sizes.

AOD sensitivity analysis was conducted on a previously inspected test split. Its protocol explicitly records this exploratory status. Validation selects checkpoints and the final candidate; all declared candidates remain available to expose sensitivity rather than only the selected outcome.

## Rights

No software or dataset license is newly assigned in this release. Consult the respective rights holders and upstream terms for reuse beyond the permissions otherwise applicable. Public availability is not a statement of unrestricted licensing.

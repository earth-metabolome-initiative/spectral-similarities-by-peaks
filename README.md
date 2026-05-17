# spectral-similarities-by-peaks

[![CI](https://github.com/earth-metabolome-initiative/spectral-similarities-by-peaks/actions/workflows/ci.yml/badge.svg)](https://github.com/earth-metabolome-initiative/spectral-similarities-by-peaks/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/earth-metabolome-initiative/spectral-similarities-by-peaks/branch/main/graph/badge.svg)](https://codecov.io/gh/earth-metabolome-initiative/spectral-similarities-by-peaks)
[![Rust 1.86+](https://img.shields.io/badge/rust-1.86%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Experiment on measuring when MS2 spectral similarity distributions stop changing as fewer or more fragment peaks are retained.

The first executable slice is a Rust CLI that:

- retrieves the harmonized annotated MS2 top-128 dataset or the GeMS-A10 top-128 dataset through `mascot-rs`.
- truncates every spectrum to configured top-intensity peak counts.
- merges peaks closer than `2 * mz_tolerance` by default, matching the well-separated precondition used by the linear/Flash similarity implementations in `mass_spectrometry`.
- computes exact top non-self neighbors with direct and modified Flash cosine or Flash entropy indexes.
- writes raw neighbor scores, per-cutoff histograms, adjacent peak-count comparisons, and full pairwise peak-count comparison grids.
- optionally scores NPC pathways by summing the selected spectrum similarity to a fixed number of pathway representatives.

Experiment runs:

Default similarity parametrization:

| CLI config | Metric | m/z exponent | Intensity exponent | Entropy weighting |
| --- | --- | ---: | ---: | --- |
| `cosine:0.0:1.0` | Direct cosine | `0.0` | `1.0` | N/A |
| `modified-cosine:0.0:1.0` | Modified cosine | `0.0` | `1.0` | N/A |
| `cosine:1.0:1.0` | Direct cosine | `1.0` | `1.0` | N/A |
| `modified-cosine:1.0:1.0` | Modified cosine | `1.0` | `1.0` | N/A |
| `cosine:0.0:0.5` | Direct cosine | `0.0` | `0.5` | N/A |
| `modified-cosine:0.0:0.5` | Modified cosine | `0.0` | `0.5` | N/A |
| `cosine:1.0:0.5` | Direct cosine | `1.0` | `0.5` | N/A |
| `modified-cosine:1.0:0.5` | Modified cosine | `1.0` | `0.5` | N/A |
| `cosine:0.0:0.25` | Direct cosine | `0.0` | `0.25` | N/A |
| `modified-cosine:0.0:0.25` | Modified cosine | `0.0` | `0.25` | N/A |
| `cosine:1.0:0.25` | Direct cosine | `1.0` | `0.25` | N/A |
| `modified-cosine:1.0:0.25` | Modified cosine | `1.0` | `0.25` | N/A |
| `cosine:3.0:0.6` | NIST-style direct cosine | `3.0` | `0.6` | N/A |
| `modified-cosine:3.0:0.6` | NIST-style modified cosine | `3.0` | `0.6` | N/A |
| `entropy:0.0:1.0:true` | Weighted entropy | `0.0` | `1.0` | `true` |
| `modified-entropy:0.0:1.0:true` | Modified weighted entropy | `0.0` | `1.0` | `true` |
| `entropy:0.0:1.0:false` | Unweighted entropy | `0.0` | `1.0` | `false` |
| `modified-entropy:0.0:1.0:false` | Modified unweighted entropy | `0.0` | `1.0` | `false` |

Harmonized full run with pathway representative scoring:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release -- scan \
  --dataset harmonized \
  --neighbors 64 \
  --mz-tolerance 0.05 \
  --pathway-representatives-per-class 5 \
  --output-dir results/harmonized-full
```

GeMS-A10 sampled run across all parts:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release -- scan \
  --dataset gems \
  --row-sample-size 100000 \
  --reference-sample-size 1000000 \
  --neighbors 64 \
  --mz-tolerance 0.05 \
  --output-dir results/gems-sampled
```

Lawrencium run:

```bash
bash slurm/lrc/setup_env.sh
bash slurm/lrc/prefetch.sh harmonized
bash slurm/lrc/submit.sh harmonized
bash slurm/lrc/status.sh harmonized 60
bash slurm/lrc/finalize.sh harmonized
bash slurm/lrc/compute_pathway_discriminability.sh harmonized
bash slurm/lrc/cancel.sh harmonized
```

```bash
bash slurm/lrc/setup_env.sh
bash slurm/lrc/prefetch.sh gems
bash slurm/lrc/submit.sh gems
bash slurm/lrc/status.sh gems 60
bash slurm/lrc/finalize.sh gems
bash slurm/lrc/cancel.sh gems
```

The Lawrencium workflow first prefetches the dataset cache, then submits `18 x 128 = 2304` restartable shard jobs. Each shard computes one `(similarity_config, retained_peak_count)` checkpoint under `distributions/<config>/top_<k>.bincode.zst`. Once the shard grid is complete, `finalize.sh` submits an 18-task array (one shard per similarity config, each running `finalize-shard` and writing to `_finalize_shards/<config>/`) followed by a single dependent merge job (`finalize-merge`) that concatenates the per-config outputs into the canonical top-level Parquet, NumPy, heatmap, and pathway artifacts.

For the AUROC / AUPRC pathway-classification metrics, `slurm/lrc/compute_pathway_discriminability.sh` submits a single job that streams `pathway_shards/<config>/top_<k>/pathway_scores.parquet` and emits `pathway_discriminability.parquet` plus `pathway_discriminability_summary.parquet`. The streaming reader processes one shard at a time so it runs comfortably on a single node even when the merged `pathway_scores.parquet` would be too large. The output is a few MB, so pulling just those two parquets back from the cluster afterwards avoids transferring the underlying hundreds of GB of pairwise scores.

All parquet artifacts produced by this binary are written with the parquet `zstd` codec by default (level 11, configured once in `crates/cli/src/output.rs::parquet_writer_props`); the dense `distribution_grid.npz` / `pathway_prediction_distribution_grid.npz` files are written with `NpzWriter::new_compressed`. Readers don't need to opt in — both parquet and `numpy.savez_compressed` consumers handle the deflated streams transparently. For artifacts that pre-date this change (e.g., the harmonized-full directory currently on Lawrencium), the `slurm/lrc/compress_parquets.sh harmonized` job runs the `re-encode-parquets` CLI subcommand, which walks every `.parquet` under the given directory and rewrites it in place using the new codec. The result stays a valid parquet (random columnar access preserved); a 10-shard sample showed an 85.8 % size reduction relative to the legacy Snappy encoding, so the subsequent rsync / Globus pull from the cluster shrinks by roughly 7×.

Use `bash slurm/lrc/cancel.sh all --include-legacy` to cancel all spectral jobs, including old generic `spectral-shard` arrays, and remove interrupted temporary checkpoint files.

Full local smoke test:

```bash
cargo test --test full_smoke
```

This runs a deterministic end-to-end synthetic scan by parsing the CLI in-process and dispatching the crate directly. It checks the generated Parquet, NumPy, SVG, and PNG artifacts plus CLI help output. The synthetic scan avoids dataset downloads while still exercising spectrum preparation, direct and modified cosine and entropy scoring, fixed reference sampling, top-k neighbor collection, distribution summaries, histograms, full comparison grids, heatmap rendering, and pathway scoring.

Outputs:

- `distribution_summary.parquet`: mean, standard deviation, and quantiles for each score distribution.
- `distribution_histograms.parquet`: fixed-width histogram bins over the `[0, 1]` score range for every distribution.
- `distribution_tests.parquet`: adjacent peak-count comparisons using two-sample KS statistic, asymptotic KS p-value, and 1D Wasserstein distance.
- `distribution_grid.parquet`: the full pairwise peak-count comparison grid as a long table.
- `distribution_grid.npz`: dense NumPy matrices shaped as `similarity_config x peak_count_a x peak_count_b` for heatmap visualization.
- `distribution_grid_configs.parquet`: config-axis metadata for `distribution_grid.npz`.
- `distributions/<config>/top_<k>.bincode.zst`: zstd-compressed serde checkpoints for sorted score distributions, reused automatically when a run is restarted with matching score-affecting arguments. Older uncompressed `.bincode` checkpoints are migrated when reused.
- `pathway_shards/<config>/top_<k>/`: per-shard pathway score and prediction Parquet files emitted by `scan-shard` and merged by `finalize-scan`.
- `heatmaps/<config>/*.svg` and `heatmaps/<config>/*.png`: static heatmaps for mean delta, KS statistic, asymptotic KS p-value, and 1D Wasserstein distance.
- `pathway_scores.parquet`: optional similarity-sum scores from each query to each NPC pathway representative group, emitted when `--pathway-representatives-per-class` is greater than zero.
- `pathway_predictions.parquet`: optional best-pathway predictions from the representative similarity sums.
- `pathway_prediction_metrics.parquet`: per-pathway one-vs-rest accuracy and MCC at every peak count, plus support-weighted averages.
- `pathway_prediction_distribution_grid.parquet`: full pairwise peak-count comparison grid for categorical prediction distributions.
- `pathway_prediction_distribution_grid.npz`: dense NumPy matrices for total variation, Jensen-Shannon distance, and Hellinger distance between prediction distributions.
- `pathway_prediction_heatmaps/<config>/*.svg` and `pathway_prediction_heatmaps/<config>/*.png`: static heatmaps for categorical prediction-distribution drift.
- `pathway_prediction_plots/<config>/*.svg` and `pathway_prediction_plots/<config>/*.png`: accuracy and MCC line plots by retained peak count, with one line per pathway and one support-weighted average line.
- `pathway_discriminability.parquet`: per-`(dataset, config, peak_count)` AUROC and AUPRC for the binary pathway-pair classifier (candidate pathway matches query pathway), plus `n_positives` and `n_negatives`.
- `pathway_discriminability_summary.parquet`: per-`(dataset, config)` mean AUROC / mean AUPRC across the peak-count grid.
- `pathway_discriminability_plots/auroc.{svg,png}` and `pathway_discriminability_plots/auprc.{svg,png}`: AUROC and AUPRC line plots by retained peak count with one line per similarity config; colour encodes the metric family and dash pattern encodes the m/z exponent.

The peak-count grid is always `1..=128`, so `distribution_grid.npz` contains full `128 x 128` matrices. `--row-sample-size` samples query rows, while `--reference-sample-size` samples the fixed reference columns used by nearest-neighbor search. The selected query and reference ids are reused across every peak count, so distribution changes are attributable to peak retention rather than changing samples.

The current distribution comparisons avoid assuming a parametric score family. The nonparametric outputs include empirical quantiles, fixed-bin histograms, two-sample KS statistic, asymptotic KS p-value, and 1D Wasserstein distance.

Existing `pathway_predictions.parquet` outputs can be reprocessed with `render-pathway-artifacts`:

```bash
target/release/spectral-similarities-by-peaks render-pathway-artifacts \
  --output-dir results/harmonized-full
```

## Pathway-classification results

Each `(config, peak_count)` cell of `pathway_scores.parquet` defines a binary classifier: the positive class is "candidate NPC pathway equals the query NPC pathway", and the similarity score itself ranks pairs. AUROC and AUPRC of that ranking measure how well the metric concentrates within-pathway similarity above cross-pathway similarity, independent of any chosen significance threshold. The first two tables report each config's **best** micro-averaged AUROC and AUPRC across the `1..=128` peak grid on the harmonized dataset, with the peak count at which each maximum is reached. Numbers come from `pathway_discriminability_summary.parquet` (`best_auroc`, `best_auroc_peak_count`, `best_auprc`, `best_auprc_peak_count`). Pooled over all `(query, candidate)` pairs the signal is weak — every config lands within 0.45–0.54 AUROC — and the best operating point lives in the 13–16 retained-peak range for the top performers.

#### Best per config — AUROC (micro-averaged)

| Family | m/z | Intensity | Weighted | Best AUROC | Top-k @ best AUROC |
| --- | ---: | ---: | :---: | ---: | ---: |
| cosine | 1.0 | 0.25 | — | 0.5387 | 16 |
| modified-cosine | 0.0 | 0.25 | — | 0.5356 | 15 |
| entropy | 0.0 | 1.00 | true | 0.5351 | 16 |
| cosine | 3.0 | 0.60 | — | 0.5345 | 14 |
| cosine | 0.0 | 0.25 | — | 0.5328 | 15 |
| modified-entropy | 0.0 | 1.00 | true | 0.5324 | 13 |
| cosine | 1.0 | 0.50 | — | 0.5286 | 14 |
| cosine | 0.0 | 0.50 | — | 0.5260 | 14 |
| modified-cosine | 0.0 | 1.00 | — | 0.5256 | 5 |
| entropy | 0.0 | 1.00 | false | 0.5243 | 14 |
| modified-cosine | 0.0 | 0.50 | — | 0.5240 | 1 |
| modified-cosine | 1.0 | 0.25 | — | 0.5240 | 1 |
| modified-cosine | 1.0 | 0.50 | — | 0.5240 | 1 |
| modified-cosine | 1.0 | 1.00 | — | 0.5240 | 1 |
| modified-cosine | 3.0 | 0.60 | — | 0.5240 | 1 |
| modified-entropy | 0.0 | 1.00 | false | 0.5240 | 1 |
| cosine | 1.0 | 1.00 | — | 0.5234 | 14 |
| cosine | 0.0 | 1.00 | — | 0.5225 | 14 |

#### Best per config — AUPRC (micro-averaged)

| Family | m/z | Intensity | Weighted | Best AUPRC | Top-k @ best AUPRC |
| --- | ---: | ---: | :---: | ---: | ---: |
| cosine | 1.0 | 0.25 | — | 0.1751 | 29 |
| cosine | 0.0 | 0.25 | — | 0.1713 | 29 |
| entropy | 0.0 | 1.00 | true | 0.1695 | 29 |
| cosine | 0.0 | 0.50 | — | 0.1555 | 28 |
| cosine | 1.0 | 0.50 | — | 0.1541 | 28 |
| entropy | 0.0 | 1.00 | false | 0.1538 | 41 |
| cosine | 3.0 | 0.60 | — | 0.1534 | 18 |
| modified-cosine | 0.0 | 0.25 | — | 0.1523 | 90 |
| modified-cosine | 0.0 | 0.50 | — | 0.1447 | 1 |
| modified-cosine | 0.0 | 1.00 | — | 0.1447 | 1 |
| modified-cosine | 1.0 | 0.25 | — | 0.1447 | 1 |
| modified-cosine | 1.0 | 0.50 | — | 0.1447 | 1 |
| modified-cosine | 1.0 | 1.00 | — | 0.1447 | 1 |
| modified-cosine | 3.0 | 0.60 | — | 0.1447 | 1 |
| modified-entropy | 0.0 | 1.00 | false | 0.1447 | 1 |
| modified-entropy | 0.0 | 1.00 | true | 0.1447 | 1 |
| cosine | 1.0 | 1.00 | — | 0.1438 | 13 |
| cosine | 0.0 | 1.00 | — | 0.1426 | 9 |

The micro-averaged view hides a much stronger per-pathway signal. Splitting the classifier into one-vs-rest classifiers per base NPC pathway (rows of `pathway_discriminability_per_class.parquet` with the corresponding query pathway as the fixed positive class) shows that each pathway has a different optimal `(config, peak_count)`, and that the per-pathway AUROC ranges all the way up to 0.68. The remaining 18 multi-pathway labels (e.g. `Alkaloids|Polyketides`) leave zero positives once the candidate pathway must match exactly and so produce NaN one-vs-rest AUROC — they're omitted.

#### Per-pathway best config — AUROC (one-vs-rest)

| Pathway | Family | m/z | Intensity | Weighted | Best AUROC | Top-k @ best AUROC |
| --- | --- | ---: | ---: | :---: | ---: | ---: |
| Terpenoids | modified-cosine | 0.0 | 0.25 | — | 0.6815 | 47 |
| Shikimates and Phenylpropanoids | cosine | 1.0 | 0.25 | — | 0.6749 | 27 |
| Polyketides | cosine | 0.0 | 1.00 | — | 0.6449 | 128 |
| Amino acids and Peptides | modified-cosine | 3.0 | 0.60 | — | 0.5969 | 128 |
| Carbohydrates | cosine | 3.0 | 0.60 | — | 0.5953 | 117 |
| Alkaloids | modified-cosine | 0.0 | 0.25 | — | 0.5311 | 1 |
| Fatty acids | modified-cosine | 3.0 | 0.60 | — | 0.5205 | 2 |

#### Per-pathway best config — AUPRC (one-vs-rest)

| Pathway | Family | m/z | Intensity | Weighted | Best AUPRC | Top-k @ best AUPRC |
| --- | --- | ---: | ---: | :---: | ---: | ---: |
| Amino acids and Peptides | modified-cosine | 3.0 | 0.60 | — | 0.4442 | 15 |
| Terpenoids | cosine | 0.0 | 0.25 | — | 0.4352 | 47 |
| Shikimates and Phenylpropanoids | cosine | 1.0 | 0.25 | — | 0.3279 | 29 |
| Polyketides | cosine | 0.0 | 1.00 | — | 0.2673 | 128 |
| Carbohydrates | cosine | 3.0 | 0.60 | — | 0.1779 | 86 |
| Alkaloids | modified-cosine | 0.0 | 0.25 | — | 0.1561 | 1 |
| Fatty acids | cosine | 1.0 | 0.50 | — | 0.1428 | 2 |

The optimal `(family, m/z, intensity)` rotates almost completely across pathways: Terpenoids and Shikimates/Phenylpropanoids prefer low-intensity-exponent direct or modified cosine and reach their plateau at top-27 to top-47; Polyketides keep improving all the way to the full top-128 on direct cosine with no intensity flattening; Amino-acids-and-Peptides and Carbohydrates lean on the NIST-style `mz³ × √int` weighting and also benefit from large peak counts; Alkaloids and Fatty acids barely separate at all and their best operating point lands at top-1 or top-2 — a top-N model is the wrong shape for those classes. The one-vs-rest plots under `pathway_discriminability_plots/per_class/<pathway>/{auroc,auprc}.{svg,png}` show the full curves.

Per-config curves of the aggregate AUROC and AUPRC by retained peak count are written to `pathway_discriminability_plots/auroc.{svg,png}` and `pathway_discriminability_plots/auprc.{svg,png}` (one line per config, colour by metric family, dash pattern by m/z exponent). The plots themselves are regenerated from the same parquets with:

```bash
target/release/spectral-similarities-by-peaks render-pathway-discriminability \
  --output-dir results/harmonized-full
```

## Web viewer

The repo also ships a small Dioxus + WebAssembly viewer that renders any of
the 8 heatmaps per config on demand in the browser. It fetches a dataset's
`distribution_grid.npz` (~9 MB) and re-uses the same `plotters` pipeline as
the CLI compiled to WASM, so changing the dataset, similarity config,
metric, or α threshold redraws an SVG client-side without any server.

The live build is deployed to GitHub Pages at
[earth-metabolome-initiative.github.io/spectral-similarities-by-peaks](https://earth-metabolome-initiative.github.io/spectral-similarities-by-peaks/).

To run it locally:

```bash
cargo install dioxus-cli --version 0.7.9 --locked  # one-off
rustup target add wasm32-unknown-unknown            # one-off
cd crates/web
dx serve --profile wasm-release --platform web
# open http://localhost:8080/spectral-similarities-by-peaks/
```

The `wasm-release` profile (defined in the workspace `Cargo.toml`) strips
DWARF debug info before `wasm-opt` runs, sidestepping the `compile unit
size was incorrect` SIGABRT seen with recent rustc + older binaryen
builds.

Committed viewer payload lives under `crates/web/public/data/`:

- `manifest.json` — list of available datasets.
- `<slug>/distribution_grid.npz` — the dense `(configs × 128 × 128)` arrays
  used at runtime.
- `<slug>/distribution_grid_configs.json` — config-axis labels (converted
  once from `distribution_grid_configs.parquet`).

To refresh the viewer with a new dataset, drop the npz into a new slug
directory and regenerate the JSON labels from the parquet (any one-off
pyarrow / arrow-rs snippet works).

## Per-config diversity ranking

The `compute-config-diversity` subcommand reduces a finished scan's
`distribution_grid.npz` to a single number per similarity configuration: the
mean of the Kolmogorov-Smirnov statistic (`D`) across every off-diagonal cell
of the 128 × 128 grid. Larger mean `D` means the score distributions shift
more across peak counts.

```bash
target/release/spectral-similarities-by-peaks compute-config-diversity \
  --output-dir results/harmonized-full
```

The table also reports the peak count at which each KS-statistic contour
(`D = 0.10`, `0.05`, `0.01`) reaches its right-edge asymptote, i.e., the
smallest retained-peak count above which the similarity-score CDF differs
from the full-peak CDF by less than that threshold. `D = 0.10` is the
data-drift literature's "small/moderate" boundary, `D = 0.05` the
"negligible/small" boundary, and `D = 0.01` a tighter reference for
near-identical distributions.

### gems-sampled

This dataset uses a 100 000-query sample searched against a 1 000 000-reference sample, both drawn from the full ~22 M-spectrum GeMS-A10 corpus (`ROW_SAMPLE_SIZE=100000`, `REFERENCE_SAMPLE_SIZE=1000000` in `slurm/lrc/submit.sh`). With 64 nearest neighbors per query that yields ~6.4 M similarity-score samples per `(config, peak_count)` cell (verified against `distribution_summary.parquet`: min 6 282 680, max 6 400 000, mean 6 399 398). Mean off-diagonal `D` ranges from 0.021 to 0.113.

| Rank | Family | m/z | Intensity | Weighted | mean D | stddev D | D = 0.10 peak | D = 0.05 peak | D = 0.01 peak |
| ---: | --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 1 | cosine | 1.0 | 0.25 | — | 0.11302 | 0.17237 | 30 | 46 | 77 |
| 2 | cosine | 0.0 | 0.25 | — | 0.10977 | 0.16755 | 30 | 47 | 75 |
| 3 | modified-cosine | 1.0 | 0.25 | — | 0.10764 | 0.17681 | 22 | 32 | 77 |
| 4 | modified-cosine | 0.0 | 0.25 | — | 0.09155 | 0.17020 | 15 | 23 | 103 |
| 5 | cosine | 3.0 | 0.60 | — | 0.08761 | 0.17140 | 18 | 30 | 60 |
| 6 | entropy | 0.0 | 1.00 | false | 0.08212 | 0.15713 | 19 | 32 | 60 |
| 7 | cosine | 0.0 | 0.50 | — | 0.08033 | 0.15775 | 17 | 30 | 59 |
| 8 | cosine | 1.0 | 0.50 | — | 0.07983 | 0.16069 | 17 | 30 | 57 |
| 9 | entropy | 0.0 | 1.00 | true | 0.07691 | 0.16142 | 14 | 29 | 59 |
| 10 | modified-entropy | 0.0 | 1.00 | false | 0.07193 | 0.14975 | 12 | 20 | 86 |
| 11 | modified-entropy | 0.0 | 1.00 | true | 0.06907 | 0.16090 | 9 | 13 | 87 |
| 12 | modified-cosine | 0.0 | 0.50 | — | 0.06420 | 0.13034 | 8 | 13 | 96 |
| 13 | modified-cosine | 1.0 | 0.50 | — | 0.05065 | 0.11160 | 9 | 14 | 69 |
| 14 | cosine | 0.0 | 1.00 | — | 0.04491 | 0.14411 | 7 | 10 | 30 |
| 15 | cosine | 1.0 | 1.00 | — | 0.04368 | 0.14481 | 7 | 9 | 29 |
| 16 | modified-cosine | 3.0 | 0.60 | — | 0.03582 | 0.07206 | 8 | 14 | 37 |
| 17 | modified-cosine | 0.0 | 1.00 | — | 0.02765 | 0.07471 | 4 | 5 | 53 |
| 18 | modified-cosine | 1.0 | 1.00 | — | 0.02126 | 0.06727 | 4 | 5 | 29 |

The intensity exponent drives most of the ranking. The four `intensity^0.25` configs (ranks 1-4) all exceed `mean D = 0.09`, while every `intensity^1.0` config sits in the bottom five with `mean D ≤ 0.045`. Taking the fourth root of intensity flattens the per-peak weights, so adding the *n*-th peak continues to move the distribution. With `intensity^1.0` the brightest one or two peaks dominate the sum and additional peaks contribute little. The `D = 0.05 peak` column reflects this: the top-4 configs reach the negligible/small boundary in the 23-47 retained-peak range, while the bottom two configs reach it at peak count 5. Modified-cosine variants rank below their direct counterparts at every intensity exponent, because shift-aware matching pulls more pairs toward the "similar" end and flattens the distribution. m/z weighting is a weaker axis: `cosine_mz0.000_int0.250` (rank 2) vs `cosine_mz1.000_int0.250` (rank 1) shows that turning m/z weighting on shifts diversity only marginally.

### harmonized-full

This is the harmonized annotated MS2 dataset with no query sampling, no reference sampling, and full top-128 truncation. All 443 905 query spectra are searched against the same pool with 64 neighbors per query, yielding ~28.4 M similarity-score samples per `(config, peak_count)` cell (verified against `distribution_summary.parquet`: min 26 847 500, max 28 409 829, mean 28 395 138, about 4.4× the gems-sampled per-cell count). The larger sample yields smoother per-peak-count CDFs, and `mean D` drops to the 0.020-0.063 range.

| Rank | Family | m/z | Intensity | Weighted | mean D | stddev D | D = 0.10 peak | D = 0.05 peak | D = 0.01 peak |
| ---: | --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 1 | cosine | 1.0 | 0.25 | — | 0.06303 | 0.15715 | 11 | 16 | 43 |
| 2 | modified-cosine | 0.0 | 0.25 | — | 0.05812 | 0.13970 | 6 | 7 | 96 |
| 3 | cosine | 0.0 | 0.25 | — | 0.05027 | 0.14653 | 7 | 10 | 61 |
| 4 | modified-cosine | 1.0 | 0.25 | — | 0.04765 | 0.13754 | 8 | 11 | 33 |
| 5 | cosine | 3.0 | 0.60 | — | 0.04723 | 0.13604 | 8 | 14 | 54 |
| 6 | modified-entropy | 0.0 | 1.00 | true | 0.04441 | 0.14659 | 6 | 8 | 46 |
| 7 | entropy | 0.0 | 1.00 | true | 0.04397 | 0.14384 | 7 | 9 | 16 |
| 8 | cosine | 1.0 | 0.50 | — | 0.03889 | 0.13495 | 6 | 9 | 23 |
| 9 | entropy | 0.0 | 1.00 | false | 0.03746 | 0.13381 | 6 | 8 | 26 |
| 10 | modified-cosine | 0.0 | 0.50 | — | 0.03581 | 0.12147 | 4 | 6 | 59 |
| 11 | cosine | 0.0 | 0.50 | — | 0.03578 | 0.13271 | 5 | 7 | 26 |
| 12 | modified-entropy | 0.0 | 1.00 | false | 0.03459 | 0.12730 | 5 | 7 | 20 |
| 13 | modified-cosine | 3.0 | 0.60 | — | 0.03251 | 0.09489 | 5 | 9 | 48 |
| 14 | modified-cosine | 1.0 | 0.50 | — | 0.03152 | 0.11399 | 5 | 7 | 21 |
| 15 | cosine | 1.0 | 1.00 | — | 0.02522 | 0.12579 | 3 | 5 | 10 |
| 16 | cosine | 0.0 | 1.00 | — | 0.02466 | 0.12480 | 3 | 4 | 9 |
| 17 | modified-cosine | 0.0 | 1.00 | — | 0.02036 | 0.10394 | 3 | 4 | 7 |
| 18 | modified-cosine | 1.0 | 1.00 | — | 0.01983 | 0.09916 | 3 | 4 | 7 |

The ordering is broadly preserved: `intensity^0.25` still leads (ranks 1-4) and `intensity^1.0` still trails (ranks 15-18), so the intensity-exponent effect is a property of the metric rather than a sampling artifact. Two differences from gems-sampled stand out. First, the top-rank `mean D` is about 44 % lower (0.063 vs 0.113 at rank 1). The gap narrows down the ranking, with the bottom-rank configs almost unchanged at 0.020 vs 0.021. With about 4.4× more score samples per cell the empirical CDFs are smoother, adjacent peak counts produce more similar distributions, and the largest pairwise D values shrink the most. Second, the `D = 0.05 peak` column is mostly single-digit: 14 of 18 configs reach the negligible/small boundary at peak count ≤ 9, and the maximum is 16 (for `cosine_mz1.000_int0.250`), well below the gems top-4 range of 23-47. On harmonized data, retaining more than 16 top peaks does not change the small-effect threshold for any config. The `D = 0.01` column is a different story: the strict boundary still requires 20-60 peaks for most configs, so the full 128 peaks are not wasted if strict equivalence matters. Modified-cosine and `intensity^1.0` configs reach the `D = 0.05` plateau at 4-5 peaks.

### Direct-vs-modified convergence asymmetry (partial pattern)

Some modified variants reach the loose thresholds `D = 0.10` and `D = 0.05` at fewer retained peaks than their direct counterparts, but reach the strict `D = 0.01` boundary at more retained peaks. The clearest examples are in gems-sampled: `cosine_mz0.000_int0.500` (17 / 30 / 59) vs `modified_cosine_mz0.000_int0.500` (8 / 13 / 96), and `entropy_mz0.000_int1.000_weightedtrue` (14 / 29 / 59) vs `modified_entropy_mz0.000_int1.000_weightedtrue` (9 / 13 / 87). Across the nine direct/modified pairs, the `D = 0.01` peak is larger for the modified variant in 6 of 9 pairs in gems-sampled and 3 of 9 in harmonized-full, with the positive cases concentrated on the `mz = 0` rows. The asymmetry is therefore selective: it appears mostly at `mz = 0` and disappears or reverses at `mz = 1` and `mz = 3`. One mechanism consistent with the positive cases is that modified variants admit a second matching channel via peaks shifted by the precursor mass difference. Once the precursor shift is fixed, a few well-placed peaks unlock most of that channel's discriminative power, so the score distribution moves quickly per added peak at low retained counts. The shifted matches continue to find marginally correlated peaks at higher counts, so the distribution stabilizes only at large peak counts and the `D = 0.01` boundary is reached late. With `mz ≠ 0` the m/z weighting damps the contribution of shifted peaks at unusual masses and suppresses this long tail. For practical use: if a small-effect peak-count plateau suffices, both variants stabilize at low peak counts. If strict score reproducibility is required and m/z weighting is off, the direct variants reach it sooner.

### What this means in practice

Higher `mean D` does not mean "better metric". It means the metric responds to more of the spectrum. For downstream tasks that can exploit fine-grained differences across peaks (such as distinguishing structurally similar molecules), a high-diversity config like `cosine_mz1.000_int0.250` carries more signal. For a metric that converges at low peak counts and is robust to spectrum truncation, the low-diversity configs (`modified_cosine_*_int1.000`) plateau at single-digit peak counts and are cheaper to compute. The rankings agree between the two datasets, so the qualitative choice generalizes. Only the absolute magnitudes shift with sample size.

As a rule of thumb across both datasets, retaining the top 50 peaks holds the score distribution within the negligible-drift threshold (D ≤ 0.05) for every config tested. Reaching the strict-equivalence threshold (D ≤ 0.01) requires up to 96 peaks on harmonized and up to 103 on gems (a single config), so retaining the full top-128 is the conservative default when exact score distributions need to be reproduced.

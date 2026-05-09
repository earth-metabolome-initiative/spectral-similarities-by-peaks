# spectral-similarities-by-peaks

[![CI](https://github.com/earth-metabolome-initiative/spectral-similarities-by-peaks/actions/workflows/ci.yml/badge.svg)](https://github.com/earth-metabolome-initiative/spectral-similarities-by-peaks/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/earth-metabolome-initiative/spectral-similarities-by-peaks/branch/main/graph/badge.svg)](https://codecov.io/gh/earth-metabolome-initiative/spectral-similarities-by-peaks)
[![Rust 1.86+](https://img.shields.io/badge/rust-1.86%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Experiment on measuring when MS2 spectral similarity distributions stop changing as fewer or more fragment peaks are retained.

The first executable slice is a Rust CLI that:

- retrieves the harmonized annotated MS2 top-128 dataset or the GeMS-A10 top-128 dataset through `mascot-rs`;
- truncates every spectrum to configured top-intensity peak counts;
- merges peaks closer than `2 * mz_tolerance` by default, matching the well-separated precondition used by the linear/Flash similarity implementations in `mass_spectrometry`;
- computes exact top non-self neighbors with direct and modified Flash cosine or Flash entropy indexes;
- writes raw neighbor scores, per-cutoff histograms, adjacent peak-count comparisons, and full pairwise peak-count comparison grids.
- optionally scores NPC pathways by summing direct or modified cosine similarity to a fixed number of pathway representatives.

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
- `distributions/<config>/top_<k>.bincode`: serde checkpoints for sorted score distributions, reused automatically when a run is restarted with matching score-affecting arguments.
- `heatmaps/<config>/*.svg` and `heatmaps/<config>/*.png`: static heatmaps for mean delta, KS statistic, asymptotic KS p-value, and 1D Wasserstein distance.
- `pathway_scores.parquet`: optional direct/modified cosine-sum scores from each query to each NPC pathway representative group, emitted when `--pathway-representatives-per-class` is greater than zero.
- `pathway_predictions.parquet`: optional best-pathway predictions from the representative direct/modified cosine sums.

The peak-count grid is always `1..=128`, so `distribution_grid.npz` contains full `128 x 128` matrices. `--row-sample-size` samples query rows, while `--reference-sample-size` samples the fixed reference columns used by nearest-neighbor search. The selected query and reference ids are reused across every peak count, so distribution changes are attributable to peak retention rather than changing samples.

The current distribution comparisons avoid assuming a parametric score family. The nonparametric outputs include empirical quantiles, fixed-bin histograms, two-sample KS statistic, asymptotic KS p-value, and 1D Wasserstein distance.

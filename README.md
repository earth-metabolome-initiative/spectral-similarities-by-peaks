# spectral-similarities-by-peaks

Experiment on measuring when MS2 spectral similarity distributions stop changing as fewer or more fragment peaks are retained.

The first executable slice is a Rust CLI that:

- retrieves the harmonized annotated MS2 top-128 dataset or the GeMS-A10 top-128 dataset through `mascot-rs`;
- truncates every spectrum to configured top-intensity peak counts;
- merges peaks closer than `2 * mz_tolerance` by default, matching the well-separated precondition used by the linear/Flash similarity implementations in `mass_spectrometry`;
- computes exact top non-self neighbors with Flash cosine or Flash entropy indexes;
- writes raw neighbor scores, per-cutoff histograms, adjacent peak-count comparisons, and full pairwise peak-count comparison grids.
- optionally scores NPC pathways by summing cosine similarity to a fixed number of pathway representatives.

Example smoke run on a small harmonized subset:

```bash
cargo run --release -- scan \
  --dataset harmonized \
  --max-spectra 1000 \
  --row-sample-size 200 \
  --reference-sample-size 1000 \
  --peak-counts 8,16,32 \
  --neighbors 10 \
  --mz-tolerance 0.05 \
  --pathway-representatives-per-class 5 \
  --output-dir results/smoke
```

Full local smoke test:

```bash
cargo test --test full_smoke
```

This runs a deterministic end-to-end synthetic scan by parsing the CLI in-process and dispatching the crate directly. It checks the generated CSV artifacts plus CLI help output. The synthetic scan avoids dataset downloads while still exercising spectrum preparation, cosine and entropy scoring, fixed reference sampling, top-k neighbor collection, distribution summaries, histograms, full comparison grids, and pathway scoring.

Full local verification:

```bash
cargo fmt --check
cargo check
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
cargo test
```

Useful GeMS iteration starts with one or a few parts:

```bash
cargo run --release -- scan \
  --dataset gems \
  --gems-parts 0 \
  --row-sample-size 10000 \
  --reference-sample-size 100000 \
  --peak-counts 8,16,32,64,128 \
  --neighbors 10 \
  --output-dir results/gems-part-0
```

Outputs:

- `similarities.csv`: one row per query, retained peak count, similarity config, and retained neighbor. Use `rank == 1` for best-neighbor-only distributions; use all ranks for the full retained-neighbor distribution.
- `distribution_summary.csv`: mean, standard deviation, and quantiles for each score distribution.
- `distribution_histograms.csv`: fixed-width histogram bins over the `[0, 1]` score range for every distribution.
- `distribution_tests.csv`: adjacent peak-count comparisons using two-sample KS statistic, asymptotic KS p-value, and 1D Wasserstein distance.
- `distribution_grid.csv`: the full pairwise peak-count comparison grid for heatmap visualization of the same distance/test columns.
- `pathway_scores.csv`: optional cosine-sum scores from each query to each NPC pathway representative group, emitted when `--pathway-representatives-per-class` is greater than zero.
- `pathway_predictions.csv`: optional best-pathway predictions from the representative cosine sums.

The default peak-count grid is `1..=128`, so `distribution_grid.csv` is a full `128 x 128` comparison grid unless `--peak-counts` narrows it. `--row-sample-size` samples query rows, while `--reference-sample-size` samples the fixed reference columns used by nearest-neighbor search. The selected query and reference ids are reused across every peak count, so distribution changes are attributable to peak retention rather than changing samples.

The current distribution comparisons avoid assuming a parametric score family. The nonparametric outputs include empirical quantiles, fixed-bin histograms, two-sample KS statistic, asymptotic KS p-value, and 1D Wasserstein distance.

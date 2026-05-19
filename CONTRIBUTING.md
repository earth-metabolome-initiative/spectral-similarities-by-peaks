# Contributing to spectral-similarities-by-peaks

The [README](README.md) keeps the science-facing summary, the headline results, and the public-facing entry points. This document covers the developer workflow.

## Local development

### Local single-process runs

```bash
# Harmonized full run with pathway representative scoring
RUSTFLAGS="-C target-cpu=native" cargo run --release -- scan \
  --dataset harmonized \
  --neighbors 64 \
  --mz-tolerance 0.05 \
  --pathway-representatives-per-class 5 \
  --output-dir results/harmonized-full
```

```bash
# GeMS-A10 sampled run across all parts
RUSTFLAGS="-C target-cpu=native" cargo run --release -- scan \
  --dataset gems \
  --row-sample-size 100000 \
  --reference-sample-size 1000000 \
  --neighbors 64 \
  --mz-tolerance 0.05 \
  --output-dir results/gems-sampled
```

`--row-sample-size` samples query rows. `--reference-sample-size` samples the fixed reference columns used by nearest-neighbor search. The selected query and reference ids are reused across every peak count, so distribution changes are attributable to peak retention rather than changing samples.

### Smoke test

```bash
cargo test --release --test full_smoke
```

A deterministic synthetic scan parsed in-process and dispatched directly through the CLI. Exercises spectrum preparation, every similarity scorer, fixed reference sampling, top-k neighbor collection, distribution summaries, histograms, full comparison grids, heatmap rendering, pathway scoring, and the generated Parquet / NumPy / SVG / PNG artifacts.

### Re-running individual stages

The pipeline is split across several subcommands so a finished scan can be reprocessed without re-running the score computation.

```bash
# Rebuild pathway-prediction summaries and per-config plots
target/release/spectral-similarities-by-peaks render-pathway-artifacts \
  --output-dir results/harmonized-full

# Re-render AUROC / AUPRC line plots from existing parquets
target/release/spectral-similarities-by-peaks render-pathway-discriminability \
  --output-dir results/harmonized-full

# Compute config-diversity ranking from distribution_grid.npz
target/release/spectral-similarities-by-peaks compute-config-diversity \
  --output-dir results/harmonized-full

# Compute AUROC / AUPRC of pathway-pair similarity scores
target/release/spectral-similarities-by-peaks compute-pathway-discriminability \
  --output-dir results/harmonized-full

# Export the WASM viewer's pathway-classification JSON
target/release/spectral-similarities-by-peaks export-pathway-discriminability-json \
  --output-dir results/harmonized-full
```

## Cluster workflow (Lawrencium)

A full scan takes ~70k compute hours, so it runs on LBL's Lawrencium cluster under SLURM. Wrappers in `slurm/lrc/` orchestrate the shard grid end-to-end.

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

`prefetch.sh` warms the dataset cache. `submit.sh` queues `18 x 128 = 2304` restartable shard jobs that each compute one `(similarity_config, retained_peak_count)` checkpoint under `distributions/<config>/top_<k>.bincode.zst`. Once the shard grid is complete, `finalize.sh` submits an 18-task array (one shard per similarity config, each running `finalize-shard` and writing to `_finalize_shards/<config>/`) followed by a single dependent merge job (`finalize-merge`) that concatenates the per-config outputs into the canonical top-level Parquet, NumPy, heatmap, and pathway artifacts.

`compute_pathway_discriminability.sh` submits a single job that streams `pathway_shards/<config>/top_<k>/pathway_scores.parquet` and emits `pathway_discriminability.parquet`, `pathway_discriminability_per_class.parquet`, and `pathway_discriminability_summary.parquet`. Per-shard streaming keeps peak memory bounded even when the merged `pathway_scores.parquet` would not fit on a single node. The outputs are a few MB, so pulling just those parquets back from the cluster avoids transferring the underlying hundreds of GB of pairwise scores.

Use `bash slurm/lrc/cancel.sh all --include-legacy` to cancel every spectral job, including legacy `spectral-shard` arrays, and remove interrupted temporary checkpoint files.

### Parquet compression

Every parquet artifact this binary writes uses the parquet `zstd` codec at level 11, configured once in `crates/cli/src/output.rs::parquet_writer_props`. The dense `distribution_grid.npz` and `pathway_prediction_distribution_grid.npz` files use `NpzWriter::new_compressed`. Both readers handle the deflated streams transparently with no opt-in.

For artifacts that pre-date this default (e.g., the harmonized-full directory on Lawrencium before the codec switch), `slurm/lrc/compress_parquets.sh harmonized` runs the `re-encode-parquets` CLI subcommand, which walks every `.parquet` under the given directory and rewrites it in place using the new codec, preserving random columnar access. A 10-shard sample showed an 85.8 % size reduction relative to the legacy Snappy encoding.

## Web viewer (local dev)

The viewer is a Dioxus + WebAssembly single-page app under `crates/web/`. To run locally:

```bash
cargo install dioxus-cli --version 0.7.9 --locked  # one-off
rustup target add wasm32-unknown-unknown            # one-off
cd crates/web
dx serve --profile wasm-release --platform web
# open http://localhost:8080/
```

The `wasm-release` profile (defined in the workspace `Cargo.toml`) strips DWARF debug info before `wasm-opt` runs, sidestepping the `compile unit size was incorrect` SIGABRT seen with recent rustc + older binaryen builds.

### Data payload

Committed viewer payload lives under `crates/web/public/data/`:

- `manifest.json`: list of available datasets.
- `<slug>/distribution_grid.npz`: the dense `(configs, 128, 128)` arrays used at runtime.
- `<slug>/distribution_grid_configs.json`: config-axis labels converted once from `distribution_grid_configs.parquet`.
- `<slug>/pathway_discriminability_lines.json`: AUROC / AUPRC / accuracy / MCC matrices per `(config, pathway, peak_count)` consumed by the Pathways tab.

To refresh the viewer with a new dataset: drop the npz into a new slug directory, regenerate the JSON labels from the parquet (any one-off pyarrow / arrow-rs snippet works), run `export-pathway-discriminability-json` to emit the pathway-classification JSON, and update `manifest.json` with the new entry.

## Output artifact reference

Files emitted by a complete `scan` / `finalize-scan` / `compute-pathway-discriminability` / `render-pathway-discriminability` pipeline:

- `distribution_summary.parquet`: mean, standard deviation, and quantiles for each score distribution.
- `distribution_histograms.parquet`: fixed-width histogram bins over the `[0, 1]` score range for every distribution.
- `distribution_tests.parquet`: adjacent peak-count comparisons using two-sample KS statistic, asymptotic KS p-value, and 1D Wasserstein distance.
- `distribution_grid.parquet`: the full pairwise peak-count comparison grid as a long table.
- `distribution_grid.npz`: dense NumPy matrices shaped as `similarity_config x peak_count_a x peak_count_b`.
- `distribution_grid_configs.parquet`: config-axis metadata for `distribution_grid.npz`.
- `distributions/<config>/top_<k>.bincode.zst`: zstd-compressed serde checkpoints for sorted score distributions, reused automatically when a run is restarted with matching score-affecting arguments. Older uncompressed `.bincode` checkpoints are migrated when reused.
- `pathway_shards/<config>/top_<k>/`: per-shard pathway score and prediction parquet files emitted by `scan-shard` and merged by `finalize-scan`.
- `heatmaps/<config>/*.{svg,png}`: static heatmaps for mean delta, KS statistic, asymptotic KS p-value, and 1D Wasserstein distance.
- `pathway_scores.parquet`: similarity-sum scores from each query to each NPC pathway representative group, emitted when `--pathway-representatives-per-class` is greater than zero.
- `pathway_predictions.parquet`: best-pathway predictions from the representative similarity sums.
- `pathway_prediction_metrics.parquet`: per-pathway one-vs-rest accuracy and MCC at every peak count, plus support-weighted averages.
- `pathway_prediction_distribution_grid.parquet`: full pairwise peak-count comparison grid for categorical prediction distributions.
- `pathway_prediction_distribution_grid.npz`: dense NumPy matrices for total variation, Jensen-Shannon distance, and Hellinger distance between prediction distributions.
- `pathway_prediction_heatmaps/<config>/*.{svg,png}`: static heatmaps for categorical prediction-distribution drift.
- `pathway_prediction_plots/<config>/*.{svg,png}`: accuracy and MCC line plots by retained peak count, one line per pathway plus one support-weighted average line.
- `pathway_discriminability.parquet`: per-`(dataset, config, peak_count)` AUROC and AUPRC for the binary pathway-pair classifier (candidate pathway matches query pathway), plus `n_positives` and `n_negatives`.
- `pathway_discriminability_per_class.parquet`: per-`(dataset, config, peak_count, pathway)` one-vs-rest AUROC and AUPRC.
- `pathway_discriminability_summary.parquet`: per-`(dataset, config)` best AUROC / best AUPRC and the peak count at which each maximum is reached.
- `pathway_discriminability_plots/{auroc,auprc}.{svg,png}` and `pathway_discriminability_plots/per_class/<pathway>/{auroc,auprc}.{svg,png}`: AUROC and AUPRC line plots by retained peak count, one line per similarity config. Colour encodes the metric family, dash pattern encodes the m/z exponent.

## Code conventions

- Run `cargo fmt --all` and `cargo clippy --locked --all-targets --workspace -- -D warnings` before opening a PR.
- The smoke test must pass: `cargo test --release --test full_smoke`.
- Workspace builds for both native and `wasm32-unknown-unknown`. Run `cargo check -p spectral-render --target wasm32-unknown-unknown` and `cargo check -p spectral-web --target wasm32-unknown-unknown` if touching code shared with the WASM viewer.
- Avoid pushing directly to `main`. Open a PR from a feature branch.

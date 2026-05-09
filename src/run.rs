//! Command execution for the experiment binary.

use std::{fs, path::Path};

use anyhow::{Context, Result, bail};
use arrow_array::{RecordBatch, StringArray, UInt64Array};
use ndarray::{Array1, Array3};
use ndarray_npy::NpzReader;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;

use crate::{
    checkpoint::{self, CheckpointBase, RunFingerprint},
    cli::{Cli, Commands, RenderHeatmapArgs, ScanArgs},
    data::load_records,
    distribution::{
        compare_distributions, histogram_sorted_distribution, summarize_sorted_distribution,
    },
    model::{LoadedRecord, ScoreDistribution, SimilarityConfig},
    neighbors::{SearchBatch, compute_neighbors},
    output::{GridArrays, OutputWriters},
    pathway::score_pathway_representatives,
    progress::{ProgressTask, ScanProgress},
    spectra::{prepare_spectra, select_query_ids, select_reference_ids},
    visualize::write_heatmaps,
};

/// Maximum top-intensity peak count evaluated by every scan.
const MAX_PEAK_COUNT: usize = 128;

/// Dispatch parsed command-line arguments to the selected command.
///
/// # Errors
///
/// Returns an error when command validation fails, input datasets cannot be
/// loaded, spectra cannot be prepared, or output artifacts cannot be
/// written.
pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Scan(args) => run_scan(args),
        Commands::RenderHeatmaps(args) => run_render_heatmaps(&args),
    }
}

/// Re-render heatmap artifacts from an existing scan output directory.
fn run_render_heatmaps(args: &RenderHeatmapArgs) -> Result<()> {
    let progress = ScanProgress::new();
    let arrays = read_grid_npz(&args.output_dir)?;
    let configs = read_grid_configs(&args.output_dir)?;
    write_heatmaps(&args.output_dir, &configs, &arrays, &progress)
}

/// Read dense grid matrices from an existing `distribution_grid.npz` artifact.
fn read_grid_npz(output_dir: &Path) -> Result<GridArrays> {
    let path = output_dir.join("distribution_grid.npz");
    let file = fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = NpzReader::new(file).with_context(|| format!("reading {}", path.display()))?;
    let peak_counts: Array1<u64> = reader.by_name("peak_counts.npy")?;
    let mean_delta: Array3<f64> = reader.by_name("mean_delta.npy")?;
    let ks_statistic: Array3<f64> = reader.by_name("ks_statistic.npy")?;
    let ks_pvalue_asymptotic: Array3<f64> = reader.by_name("ks_pvalue_asymptotic.npy")?;
    let wasserstein_1d: Array3<f64> = reader.by_name("wasserstein_1d.npy")?;
    Ok(GridArrays {
        peak_counts,
        mean_delta,
        ks_statistic,
        ks_pvalue_asymptotic,
        wasserstein_1d,
    })
}

/// Read heatmap config labels from `distribution_grid_configs.parquet`.
fn read_grid_configs(output_dir: &Path) -> Result<Vec<String>> {
    let path = output_dir.join("distribution_grid_configs.parquet");
    let file = fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading metadata from {}", path.display()))?
        .build()
        .with_context(|| format!("building reader for {}", path.display()))?;

    let mut configs = Vec::<(usize, String)>::new();
    for batch in reader {
        read_grid_config_batch(&batch?, &mut configs)?;
    }
    configs.sort_by_key(|(config_index, _config)| *config_index);
    let configs = configs
        .into_iter()
        .map(|(_config_index, config)| config)
        .collect::<Vec<_>>();
    if configs.is_empty() {
        bail!("{} contains no heatmap configs", path.display());
    }
    Ok(configs)
}

/// Append grid config rows from one `Arrow` record batch.
fn read_grid_config_batch(batch: &RecordBatch, configs: &mut Vec<(usize, String)>) -> Result<()> {
    let config_indices = required_column::<UInt64Array>(batch, "config_index")?;
    let config_names = required_column::<StringArray>(batch, "config")?;
    for row in 0..batch.num_rows() {
        let config_index = usize::try_from(config_indices.value(row))
            .context("config_index does not fit usize")?;
        configs.push((config_index, config_names.value(row).to_string()));
    }
    Ok(())
}

/// Return a typed required `Arrow` column from a record batch.
fn required_column<'a, ArrayType: 'static>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a ArrayType> {
    batch
        .column_by_name(name)
        .with_context(|| format!("missing {name} column"))?
        .as_any()
        .downcast_ref::<ArrayType>()
        .with_context(|| format!("{name} column has unexpected type"))
}

/// Execute a full similarity scan and write all artifacts.
fn run_scan(mut args: ScanArgs) -> Result<()> {
    args.validate()?;
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("creating {}", args.output_dir.display()))?;

    let progress = ScanProgress::new();
    let load_progress = progress.spinner(format!("loading {} records", args.dataset.as_str()));
    let mut records = load_records(&args)?;
    load_progress.finish();
    if let Some(max_spectra) = args.max_spectra {
        records.truncate(max_spectra.min(records.len()));
    }
    if records.is_empty() {
        bail!("no spectra loaded");
    }

    let query_ids = select_query_ids(records.len(), args.row_sample_size, args.seed);
    let reference_ids = select_reference_ids(records.len(), args.reference_sample_size, args.seed);
    let checkpoint_base = CheckpointBase::new(&args, &records, &query_ids, &reference_ids);
    let writer_progress = progress.spinner("opening output writers");
    let mut writers = OutputWriters::create(&args.output_dir)?;
    writer_progress.finish();
    let total_peak_runs = args.similarity_configs.len().saturating_mul(MAX_PEAK_COUNT);
    let scan_progress = progress.bar(
        u64::try_from(total_peak_runs).unwrap_or(u64::MAX),
        "scanning peak-count grid",
    );

    let inputs = ScanInputs {
        args: &args,
        progress: &progress,
        records: &records,
        query_ids: &query_ids,
        reference_ids: &reference_ids,
        checkpoint_base: &checkpoint_base,
        scan_progress: &scan_progress,
    };
    for config in &args.similarity_configs {
        run_similarity_config(&inputs, &mut writers, config)?;
    }
    scan_progress.finish();

    writers.finish(&progress)
}

/// Shared immutable inputs for one scan.
struct ScanInputs<'a> {
    /// Parsed scan arguments.
    args: &'a ScanArgs,
    /// Shared progress reporter.
    progress: &'a ScanProgress,
    /// Loaded spectrum records.
    records: &'a [LoadedRecord],
    /// Query row ids selected for the run.
    query_ids: &'a [usize],
    /// Reference row ids selected for the run.
    reference_ids: &'a [usize],
    /// Shared checkpoint fingerprint fields.
    checkpoint_base: &'a CheckpointBase,
    /// Outer peak-grid progress bar.
    scan_progress: &'a ProgressTask,
}

/// Run all peak counts and distribution comparisons for one similarity config.
fn run_similarity_config(
    inputs: &ScanInputs<'_>,
    writers: &mut OutputWriters,
    config: &SimilarityConfig,
) -> Result<()> {
    let config_name = config.name();
    let fingerprint = inputs
        .checkpoint_base
        .fingerprint(inputs.args, config, &config_name);
    let mut distributions = Vec::with_capacity(MAX_PEAK_COUNT);
    for peak_count in 1..=MAX_PEAK_COUNT {
        let distribution = run_peak_count(
            inputs,
            writers,
            config,
            &config_name,
            &fingerprint,
            peak_count,
        )?;
        distributions.push(distribution);
        inputs.scan_progress.inc(1);
    }

    write_adjacent_comparisons(inputs, writers, config, &config_name, &distributions)?;
    write_grid_comparisons(inputs, writers, config, &config_name, &distributions)
}

/// Run one retained-peak count for one similarity config.
fn run_peak_count(
    inputs: &ScanInputs<'_>,
    writers: &mut OutputWriters,
    config: &SimilarityConfig,
    config_name: &str,
    fingerprint: &RunFingerprint,
    peak_count: usize,
) -> Result<ScoreDistribution> {
    inputs
        .scan_progress
        .set_message(format!("scanning {config_name} top {peak_count} peaks"));
    if let Some(distribution) = checkpoint::load_distribution(
        &inputs.args.output_dir,
        inputs.args.dataset.as_str(),
        config_name,
        config.metric_label(),
        peak_count,
        fingerprint,
    ) {
        inputs
            .scan_progress
            .set_message(format!("using cached {config_name} top {peak_count} peaks"));
        write_cached_distribution_outputs(inputs, writers, config, &distribution).with_context(
            || format!("writing cached outputs for {config_name} top {peak_count} peaks"),
        )?;
        return Ok(distribution);
    }

    let spectra = prepare_spectra(
        inputs.progress,
        inputs.records,
        peak_count,
        inputs.args.mz_tolerance,
        !inputs.args.no_merge_close_peaks,
    )
    .with_context(|| format!("preparing spectra for top {peak_count} peaks"))?;

    let hits = compute_neighbors(&SearchBatch {
        args: inputs.args,
        progress: inputs.progress,
        config,
        peak_count,
        records: inputs.records,
        spectra: &spectra,
        query_ids: inputs.query_ids,
        reference_ids: inputs.reference_ids,
    })
    .with_context(|| format!("computing {config_name} neighbors for top {peak_count} peaks"))?;

    let mut scores = hits.into_iter().map(|hit| hit.score).collect::<Vec<_>>();
    scores.par_sort_by(f64::total_cmp);

    let summary = summarize_sorted_distribution(inputs.args, config, peak_count, &scores)?;
    let distribution = ScoreDistribution {
        peak_count,
        scores,
        mean: summary.mean,
    };
    checkpoint::store_distribution(
        &inputs.args.output_dir,
        inputs.args.dataset.as_str(),
        config_name,
        config.metric_label(),
        &distribution,
        fingerprint,
    )
    .with_context(|| format!("storing {config_name} top {peak_count} distribution checkpoint"))?;
    let histogram =
        histogram_sorted_distribution(inputs.args, config, peak_count, &distribution.scores)?;
    writers.write_histogram(&histogram)?;

    if let Some((pathway_scores, pathway_predictions)) = score_pathway_representatives(
        inputs.args,
        inputs.progress,
        config,
        peak_count,
        inputs.records,
        &spectra,
        inputs.query_ids,
    )? {
        writers.write_pathway_scores(&pathway_scores)?;
        writers.write_pathway_predictions(&pathway_predictions)?;
    }

    writers.write_summary(&summary)?;
    Ok(distribution)
}

/// Write regenerated artifacts for a cached score distribution.
fn write_cached_distribution_outputs(
    inputs: &ScanInputs<'_>,
    writers: &mut OutputWriters,
    config: &SimilarityConfig,
    distribution: &ScoreDistribution,
) -> Result<()> {
    let peak_count = distribution.peak_count;
    let summary =
        summarize_sorted_distribution(inputs.args, config, peak_count, &distribution.scores)?;
    let histogram =
        histogram_sorted_distribution(inputs.args, config, peak_count, &distribution.scores)?;
    writers.write_histogram(&histogram)?;

    if inputs.args.pathway_representatives_per_class > 0 && config.metric.is_cosine_family() {
        let spectra = prepare_spectra(
            inputs.progress,
            inputs.records,
            peak_count,
            inputs.args.mz_tolerance,
            !inputs.args.no_merge_close_peaks,
        )
        .with_context(|| {
            format!("preparing spectra for cached pathway scoring for top {peak_count} peaks")
        })?;
        if let Some((pathway_scores, pathway_predictions)) = score_pathway_representatives(
            inputs.args,
            inputs.progress,
            config,
            peak_count,
            inputs.records,
            &spectra,
            inputs.query_ids,
        )? {
            writers.write_pathway_scores(&pathway_scores)?;
            writers.write_pathway_predictions(&pathway_predictions)?;
        }
    }

    writers.write_summary(&summary)?;
    Ok(())
}

/// Write adjacent peak-count comparison rows for one similarity config.
fn write_adjacent_comparisons(
    inputs: &ScanInputs<'_>,
    writers: &mut OutputWriters,
    config: &SimilarityConfig,
    config_name: &str,
    distributions: &[ScoreDistribution],
) -> Result<()> {
    let adjacent_progress = inputs.progress.bar(
        u64::try_from(MAX_PEAK_COUNT.saturating_sub(1)).unwrap_or(u64::MAX),
        format!("comparing adjacent distributions for {config_name}"),
    );
    let comparisons = (0..distributions.len().saturating_sub(1))
        .into_par_iter()
        .map(|index| {
            let comparison = compare_distributions(
                inputs.args,
                config,
                &distributions[index],
                &distributions[index + 1],
            );
            adjacent_progress.inc(1);
            comparison
        })
        .collect::<Result<Vec<_>>>()?;
    for comparison in comparisons {
        writers.write_adjacent_comparison(comparison);
    }
    adjacent_progress.finish();
    writers.flush_adjacent_comparisons()
}

/// Write full peak-count grid comparison rows for one similarity config.
fn write_grid_comparisons(
    inputs: &ScanInputs<'_>,
    writers: &mut OutputWriters,
    config: &SimilarityConfig,
    config_name: &str,
    distributions: &[ScoreDistribution],
) -> Result<()> {
    let grid_progress = inputs.progress.bar(
        u64::try_from(MAX_PEAK_COUNT.saturating_mul(MAX_PEAK_COUNT)).unwrap_or(u64::MAX),
        format!("comparing full distribution grid for {config_name}"),
    );
    let width = distributions.len();
    let comparisons = (0..width.saturating_mul(width))
        .into_par_iter()
        .map(|cell| {
            let row = cell / width;
            let column = cell % width;
            let comparison = compare_distributions(
                inputs.args,
                config,
                &distributions[row],
                &distributions[column],
            );
            grid_progress.inc(1);
            comparison
        })
        .collect::<Result<Vec<_>>>()?;
    for comparison in comparisons {
        writers.write_grid_comparison(comparison);
    }
    grid_progress.finish();
    writers.flush_grid_comparisons()
}

#[cfg(test)]
/// Unit tests for resumable scan orchestration.
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    use anyhow::{Context, Result};
    use clap::Parser;

    use crate::{
        checkpoint::{self, CheckpointBase},
        cli::{Cli, Commands},
        data::load_records,
        model::ScoreDistribution,
        spectra::{select_query_ids, select_reference_ids},
    };

    use super::run_scan;

    #[test]
    /// A valid cached distribution is reused while missing peak counts are computed.
    fn scan_reuses_valid_distribution_checkpoint() -> Result<()> {
        let root = temp_root("resume")?;
        let data_dir = root.join("data");
        let output_dir = root.join("out");
        fs::create_dir_all(&data_dir)?;
        fs::create_dir_all(&output_dir)?;

        let mut args = scan_args(&data_dir, &output_dir)?;
        args.validate()?;
        let mut records = load_records(&args)?;
        if let Some(max_spectra) = args.max_spectra {
            records.truncate(max_spectra.min(records.len()));
        }
        let query_ids = select_query_ids(records.len(), args.row_sample_size, args.seed);
        let reference_ids =
            select_reference_ids(records.len(), args.reference_sample_size, args.seed);
        let config = args
            .similarity_configs
            .first()
            .context("test scan must include one similarity config")?;
        let config_name = config.name();
        let checkpoint_base = CheckpointBase::new(&args, &records, &query_ids, &reference_ids);
        let fingerprint = checkpoint_base.fingerprint(&args, config, &config_name);
        let cached = ScoreDistribution {
            peak_count: 1,
            scores: vec![0.123_456, 0.654_321, 0.777_777],
            mean: 0.518_518,
        };
        checkpoint::store_distribution(
            &output_dir,
            args.dataset.as_str(),
            &config_name,
            config.metric_label(),
            &cached,
            &fingerprint,
        )?;

        run_scan(args)?;

        let retained = checkpoint::load_distribution(
            &output_dir,
            "synthetic-smoke",
            &config_name,
            "cosine",
            1,
            &fingerprint,
        );
        assert_eq!(retained, Some(cached));
        assert!(
            checkpoint::checkpoint_path(&output_dir, &config_name, 2).exists(),
            "missing peak counts should be computed and checkpointed"
        );

        fs::remove_dir_all(root)?;
        Ok(())
    }

    /// Parse the synthetic scan arguments used by the resume test.
    fn scan_args(data_dir: &Path, output_dir: &Path) -> Result<crate::cli::ScanArgs> {
        let cli = Cli::try_parse_from([
            "spectral-similarities-by-peaks",
            "scan",
            "--dataset",
            "synthetic-smoke",
            "--data-dir",
            data_dir
                .to_str()
                .context("temporary data path is not valid UTF-8")?,
            "--output-dir",
            output_dir
                .to_str()
                .context("temporary output path is not valid UTF-8")?,
            "--similarity-config",
            "cosine:0.0:1.0",
            "--neighbors",
            "2",
            "--mz-tolerance",
            "0.05",
            "--histogram-bins",
            "4",
            "--row-sample-size",
            "4",
            "--reference-sample-size",
            "6",
            "--max-spectra",
            "8",
            "--seed",
            "99",
        ])?;
        match cli.command {
            Commands::Scan(args) => Ok(args),
            Commands::RenderHeatmaps(_args) => anyhow::bail!("expected scan command"),
        }
    }

    /// Return a unique temporary directory for one scan test.
    fn temp_root(label: &str) -> Result<PathBuf> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?;
        Ok(std::env::temp_dir().join(format!(
            "spectral-similarities-run-{label}-{}-{}",
            std::process::id(),
            timestamp.as_nanos()
        )))
    }
}

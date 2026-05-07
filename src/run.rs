//! Command execution for the experiment binary.

use std::fs;

use anyhow::{Context, Result, bail};

use crate::{
    cli::{Cli, Commands, ScanArgs},
    data::load_records,
    distribution::{compare_distributions, histogram_distribution, summarize_distribution},
    model::{LoadedRecord, ScoreDistribution, SimilarityConfig},
    neighbors::{SearchBatch, compute_neighbors},
    output::OutputWriters,
    pathway::score_pathway_representatives,
    progress::{ProgressTask, ScanProgress},
    spectra::{prepare_spectra, select_query_ids, select_reference_ids},
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
    }
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
    let mut distributions = Vec::with_capacity(MAX_PEAK_COUNT);
    for peak_count in 1..=MAX_PEAK_COUNT {
        let distribution = run_peak_count(inputs, writers, config, &config_name, peak_count)?;
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
    peak_count: usize,
) -> Result<ScoreDistribution> {
    inputs
        .scan_progress
        .set_message(format!("scanning {config_name} top {peak_count} peaks"));
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

    let scores = hits.iter().map(|hit| hit.score).collect::<Vec<_>>();
    writers.write_neighbors(&hits)?;

    let summary = summarize_distribution(inputs.args, config, peak_count, &scores)?;
    let histogram = histogram_distribution(inputs.args, config, peak_count, &scores)?;
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
    Ok(ScoreDistribution {
        peak_count,
        scores,
        mean: summary.mean,
    })
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
    for pair in distributions.windows(2) {
        let comparison = compare_distributions(inputs.args, config, &pair[0], &pair[1])?;
        writers.write_adjacent_comparison(comparison);
        adjacent_progress.inc(1);
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
    for first in distributions {
        for second in distributions {
            let comparison = compare_distributions(inputs.args, config, first, second)?;
            writers.write_grid_comparison(comparison);
            grid_progress.inc(1);
        }
    }
    grid_progress.finish();
    writers.flush_grid_comparisons()
}

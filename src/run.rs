//! Command execution for the experiment binary.

use std::fs;

use anyhow::{Context, Result, bail};

use crate::{
    cli::{Cli, Commands, ScanArgs},
    data::load_records,
    distribution::{compare_distributions, histogram_distribution, summarize_distribution},
    model::ScoreDistribution,
    neighbors::compute_neighbors,
    output::OutputWriters,
    pathway::score_pathway_representatives,
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

    let mut records = load_records(&args)?;
    if let Some(max_spectra) = args.max_spectra {
        records.truncate(max_spectra.min(records.len()));
    }
    if records.is_empty() {
        bail!("no spectra loaded");
    }

    let query_ids = select_query_ids(records.len(), args.row_sample_size, args.seed);
    let reference_ids = select_reference_ids(records.len(), args.reference_sample_size, args.seed);
    let mut writers = OutputWriters::create(&args.output_dir)?;

    for config in &args.similarity_configs {
        let mut distributions = Vec::with_capacity(MAX_PEAK_COUNT);
        for peak_count in 1..=MAX_PEAK_COUNT {
            let spectra = prepare_spectra(
                &records,
                peak_count,
                args.mz_tolerance,
                !args.no_merge_close_peaks,
            )
            .with_context(|| format!("preparing spectra for top {peak_count} peaks"))?;

            let hits = compute_neighbors(
                &args,
                config,
                peak_count,
                &records,
                &spectra,
                &query_ids,
                &reference_ids,
            )
            .with_context(|| {
                format!(
                    "computing {} neighbors for top {peak_count} peaks",
                    config.name()
                )
            })?;

            let scores = hits.iter().map(|hit| hit.score).collect::<Vec<_>>();
            writers.write_neighbors(&hits)?;

            let summary = summarize_distribution(&args, config, peak_count, &scores)?;
            let histogram = histogram_distribution(&args, config, peak_count, &scores)?;
            writers.write_histogram(&histogram)?;

            if let Some((pathway_scores, pathway_predictions)) = score_pathway_representatives(
                &args, config, peak_count, &records, &spectra, &query_ids,
            )? {
                writers.write_pathway_scores(&pathway_scores)?;
                writers.write_pathway_predictions(&pathway_predictions)?;
            }

            let current_distribution = ScoreDistribution {
                peak_count,
                scores,
                mean: summary.mean,
            };
            writers.write_summary(&summary)?;
            distributions.push(current_distribution);
        }

        for pair in distributions.windows(2) {
            let comparison = compare_distributions(&args, config, &pair[0], &pair[1])?;
            writers.write_adjacent_comparison(comparison);
        }
        writers.flush_adjacent_comparisons()?;

        for first in &distributions {
            for second in &distributions {
                let comparison = compare_distributions(&args, config, first, second)?;
                writers.write_grid_comparison(comparison);
            }
        }
        writers.flush_grid_comparisons()?;
    }

    writers.finish()
}

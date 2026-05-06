//! Command execution for the experiment binary.

use std::{fs, path::Path};

use anyhow::{Context, Result, bail};

use crate::{
    cli::{Cli, Commands, ScanArgs},
    data::load_records,
    distribution::{compare_distributions, histogram_distribution, summarize_distribution},
    model::{
        DistributionComparison, DistributionHistogramBin, DistributionSummary, NeighborHit,
        PathwayPrediction, PathwayScore, ScoreDistribution,
    },
    neighbors::compute_neighbors,
    pathway::score_pathway_representatives,
    spectra::{prepare_spectra, select_query_ids, select_reference_ids},
};

/// Dispatch parsed command-line arguments to the selected command.
///
/// # Errors
///
/// Returns an error when command validation fails, input datasets cannot be
/// loaded, spectra cannot be prepared, or output CSV artifacts cannot be
/// written.
pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Scan(args) => run_scan(args),
    }
}

/// Execute a full similarity scan and write all CSV artifacts.
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
        let mut distributions = Vec::with_capacity(args.peak_counts.len());
        for &peak_count in &args.peak_counts {
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
            writers.write_neighbors(hits)?;

            let summary = summarize_distribution(&args, config, peak_count, &scores)?;
            writers.write_histogram(histogram_distribution(&args, config, peak_count, &scores)?)?;

            if let Some((pathway_scores, pathway_predictions)) = score_pathway_representatives(
                &args, config, peak_count, &records, &spectra, &query_ids,
            )? {
                writers.write_pathway_scores(pathway_scores)?;
                writers.write_pathway_predictions(pathway_predictions)?;
            }

            let current_distribution = ScoreDistribution {
                peak_count,
                scores,
                mean: summary.mean,
            };
            writers.write_summary(summary)?;
            distributions.push(current_distribution);
        }

        for pair in distributions.windows(2) {
            let comparison = compare_distributions(&args, config, &pair[0], &pair[1])?;
            writers.write_adjacent_comparison(comparison)?;
        }
        writers.flush_adjacent_comparisons()?;

        for first in &distributions {
            for second in &distributions {
                let comparison = compare_distributions(&args, config, first, second)?;
                writers.write_grid_comparison(comparison)?;
            }
        }
        writers.flush_grid_comparisons()?;
    }

    Ok(())
}

/// CSV writers for all scan artifacts.
struct OutputWriters {
    /// Raw top-neighbor output writer.
    similarity: csv::Writer<fs::File>,
    /// Distribution summary output writer.
    summary: csv::Writer<fs::File>,
    /// Histogram output writer.
    histogram: csv::Writer<fs::File>,
    /// Adjacent-comparison output writer.
    adjacent_comparison: csv::Writer<fs::File>,
    /// Full comparison-grid output writer.
    grid_comparison: csv::Writer<fs::File>,
    /// Pathway score output writer.
    pathway_score: csv::Writer<fs::File>,
    /// Pathway prediction output writer.
    pathway_prediction: csv::Writer<fs::File>,
}

impl OutputWriters {
    /// Create every output writer under the scan output directory.
    fn create(output_dir: &Path) -> Result<Self> {
        Ok(Self {
            similarity: csv_writer(output_dir, "similarities.csv")?,
            summary: csv_writer(output_dir, "distribution_summary.csv")?,
            histogram: csv_writer(output_dir, "distribution_histograms.csv")?,
            adjacent_comparison: csv_writer(output_dir, "distribution_tests.csv")?,
            grid_comparison: csv_writer(output_dir, "distribution_grid.csv")?,
            pathway_score: csv_writer(output_dir, "pathway_scores.csv")?,
            pathway_prediction: csv_writer(output_dir, "pathway_predictions.csv")?,
        })
    }

    /// Write raw neighbor rows.
    fn write_neighbors(&mut self, hits: Vec<NeighborHit>) -> Result<()> {
        for hit in hits {
            self.similarity.serialize(hit)?;
        }
        self.similarity.flush()?;
        Ok(())
    }

    /// Write one distribution summary row.
    fn write_summary(&mut self, summary: DistributionSummary) -> Result<()> {
        self.summary.serialize(summary)?;
        self.summary.flush()?;
        Ok(())
    }

    /// Write histogram rows for one distribution.
    fn write_histogram(&mut self, bins: Vec<DistributionHistogramBin>) -> Result<()> {
        for bin in bins {
            self.histogram.serialize(bin)?;
        }
        self.histogram.flush()?;
        Ok(())
    }

    /// Write pathway score rows for one peak-count run.
    fn write_pathway_scores(&mut self, scores: Vec<PathwayScore>) -> Result<()> {
        for score in scores {
            self.pathway_score.serialize(score)?;
        }
        self.pathway_score.flush()?;
        Ok(())
    }

    /// Write pathway prediction rows for one peak-count run.
    fn write_pathway_predictions(&mut self, predictions: Vec<PathwayPrediction>) -> Result<()> {
        for prediction in predictions {
            self.pathway_prediction.serialize(prediction)?;
        }
        self.pathway_prediction.flush()?;
        Ok(())
    }

    /// Write one adjacent-distribution comparison.
    fn write_adjacent_comparison(&mut self, comparison: DistributionComparison) -> Result<()> {
        self.adjacent_comparison.serialize(comparison)?;
        Ok(())
    }

    /// Flush adjacent-distribution comparisons.
    fn flush_adjacent_comparisons(&mut self) -> Result<()> {
        self.adjacent_comparison.flush()?;
        Ok(())
    }

    /// Write one full-grid distribution comparison.
    fn write_grid_comparison(&mut self, comparison: DistributionComparison) -> Result<()> {
        self.grid_comparison.serialize(comparison)?;
        Ok(())
    }

    /// Flush full-grid distribution comparisons.
    fn flush_grid_comparisons(&mut self) -> Result<()> {
        self.grid_comparison.flush()?;
        Ok(())
    }
}

/// Create one named CSV writer under an output directory.
fn csv_writer(output_dir: &Path, file_name: &str) -> Result<csv::Writer<fs::File>> {
    let path = output_dir.join(file_name);
    csv::Writer::from_path(&path).with_context(|| format!("creating {}", path.display()))
}

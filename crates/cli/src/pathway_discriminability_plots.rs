//! Line plots of AUROC / AUPRC pathway-pair discriminability per config.
//!
//! Consumes `pathway_discriminability.parquet` (the micro-averaged
//! aggregate over every `(query, candidate)` pair, pooled per
//! `(dataset, config, peak_count)`) and
//! `pathway_discriminability_per_class.parquet` (the one-vs-rest split
//! per `(dataset, config, peak_count, pathway)`) produced by
//! `compute-pathway-discriminability`, and emits two artifact families
//! under the output directory's `pathway_discriminability_plots/`
//! subdirectory:
//!
//! - `auroc.{svg,png}` and `auprc.{svg,png}` at the subdirectory root.
//!   The pooled (micro-averaged) classifier across all queries.
//! - `per_class/<pathway_slug>/{auroc,auprc}.{svg,png}`. The per-pathway
//!   one-vs-rest curves, one chart family per pathway.
//!
//! Each chart draws one line per similarity config. Colour encodes the
//! metric family, dash pattern encodes the m/z exponent, and a colour mix
//! factor distinguishes the intensity exponent or entropy weighting within
//! a family + dash combination. The visual encoding lives in
//! `spectral_render::pathway_lines` so the CLI plots and the WASM viewer's
//! pathway tab stay in lockstep.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, Result};
use arrow_array::{Array, Float64Array, RecordBatch, StringArray, UInt64Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use plotters::prelude::{BitMapBackend, IntoDrawingArea, SVGBackend};
use spectral_render::{
    PathwayFamily, PathwayLineSeries, PathwayMetric, draw_pathway_lines, pretty_series_label,
};

use crate::{
    pathway_artifacts::{LINE_PLOT_HEIGHT, LINE_PLOT_WIDTH},
    progress::ScanProgress,
    visualize::{ensure_heatmap_font, sanitize_path_component},
};

/// Subdirectory holding the rendered discriminability plots.
const PLOTS_SUBDIR: &str = "pathway_discriminability_plots";

/// Read `pathway_discriminability.parquet` and
/// `pathway_discriminability_per_class.parquet` and write per-metric line
/// plots: a micro-averaged aggregate at the root of the plots directory
/// plus one one-vs-rest chart pair per pathway under `per_class/`.
///
/// Silently returns `Ok(())` if both parquets are missing or empty (these
/// states correspond to datasets without pathway annotations such as
/// gems-sampled, and should not abort the downstream pipeline).
///
/// # Errors
///
/// Returns an error if a parquet exists but cannot be read, or if a plot
/// cannot be written to disk.
pub fn write_pathway_discriminability_plots(
    output_dir: &Path,
    progress: &ScanProgress,
) -> Result<()> {
    ensure_heatmap_font()?;
    let plots_dir = output_dir.join(PLOTS_SUBDIR);

    let aggregate_path = output_dir.join("pathway_discriminability.parquet");
    if aggregate_path.is_file() {
        let read_progress = progress.spinner("reading pathway_discriminability.parquet");
        let rows = read_aggregate_rows(&aggregate_path)?;
        read_progress.finish();
        if !rows.is_empty() {
            fs::create_dir_all(&plots_dir)
                .with_context(|| format!("creating {}", plots_dir.display()))?;
            let render_progress =
                progress.spinner("rendering aggregate (micro-averaged) discriminability plots");
            for metric in [PathwayMetric::Auroc, PathwayMetric::Auprc] {
                let caption = format!("{} (micro-averaged)", metric.title());
                let series = build_series_from_aggregate(&rows, metric);
                let stem = plots_dir.join(metric_file_stem(metric));
                write_plot_svg(&stem.with_extension("svg"), metric, &caption, &series)?;
                write_plot_png(&stem.with_extension("png"), metric, &caption, &series)?;
            }
            render_progress.finish();
        }
    }

    let per_class_path = output_dir.join("pathway_discriminability_per_class.parquet");
    if per_class_path.is_file() {
        let read_progress = progress.spinner("reading pathway_discriminability_per_class.parquet");
        let per_class = read_per_class_rows(&per_class_path)?;
        read_progress.finish();
        if !per_class.is_empty() {
            fs::create_dir_all(&plots_dir)
                .with_context(|| format!("creating {}", plots_dir.display()))?;
            let per_class_dir = plots_dir.join("per_class");
            fs::create_dir_all(&per_class_dir)
                .with_context(|| format!("creating {}", per_class_dir.display()))?;
            let pathways = group_by_pathway(&per_class);
            let render_progress = progress.bar(
                u64::try_from(pathways.len()).unwrap_or(u64::MAX),
                "rendering per-class discriminability plots",
            );
            for (pathway, rows) in pathways {
                let subdir = per_class_dir.join(sanitize_path_component(&pathway));
                fs::create_dir_all(&subdir)
                    .with_context(|| format!("creating {}", subdir.display()))?;
                for metric in [PathwayMetric::Auroc, PathwayMetric::Auprc] {
                    let caption = format!("{}. One-vs-rest, pathway = {}", metric.title(), pathway);
                    let series = build_series_from_aggregate(&rows, metric);
                    let stem = subdir.join(metric_file_stem(metric));
                    write_plot_svg(&stem.with_extension("svg"), metric, &caption, &series)?;
                    write_plot_png(&stem.with_extension("png"), metric, &caption, &series)?;
                }
                render_progress.inc(1);
            }
            render_progress.finish();
        }
    }

    Ok(())
}

/// File-stem fragment chosen per metric, used for both the SVG and PNG outputs.
const fn metric_file_stem(metric: PathwayMetric) -> &'static str {
    match metric {
        PathwayMetric::Auroc => "auroc",
        PathwayMetric::Auprc => "auprc",
        PathwayMetric::Accuracy => "accuracy",
        PathwayMetric::Mcc => "mcc",
    }
}

/// One row of `pathway_discriminability.parquet`.
struct DiscriminabilityRow {
    /// Similarity-config slug.
    config: String,
    /// Number of retained top-intensity peaks for this row.
    peak_count: u64,
    /// Area under the ROC curve.
    auroc: f64,
    /// Area under the precision-recall curve.
    auprc: f64,
}

/// One row of `pathway_discriminability_per_class.parquet`.
struct PerClassPlotRow {
    /// Similarity-config slug.
    config: String,
    /// Number of retained top-intensity peaks for this row.
    peak_count: u64,
    /// Positive-class pathway label for this row.
    pathway: String,
    /// One-vs-rest area under the ROC curve.
    auroc: f64,
    /// One-vs-rest area under the precision-recall curve.
    auprc: f64,
}

/// Group per-class rows by pathway. Returns an ordered map keyed by pathway
/// label so per-pathway plot directories appear in stable order.
fn group_by_pathway(rows: &[PerClassPlotRow]) -> BTreeMap<String, Vec<DiscriminabilityRow>> {
    let mut grouped: BTreeMap<String, Vec<DiscriminabilityRow>> = BTreeMap::new();
    for row in rows {
        grouped
            .entry(row.pathway.clone())
            .or_default()
            .push(DiscriminabilityRow {
                config: row.config.clone(),
                peak_count: row.peak_count,
                auroc: row.auroc,
                auprc: row.auprc,
            });
    }
    grouped
}

/// Decoded config slug used to feed the shared render crate.
pub struct ParsedConfig {
    /// Similarity family.
    pub family: PathwayFamily,
    /// m/z exponent.
    pub mz: f64,
    /// Intensity exponent.
    pub intensity: f64,
    /// Optional entropy weighting flag.
    pub weighted: Option<bool>,
}

/// Parse a config slug into its visual-encoding components. Unknown slugs
/// fall back to the cosine family with neutral parameters.
pub fn parse_config_slug(slug: &str) -> ParsedConfig {
    let (family, rest) = match slug {
        s if s.starts_with("modified_cosine_") => (
            PathwayFamily::ModifiedCosine,
            &s["modified_cosine_".len()..],
        ),
        s if s.starts_with("cosine_") => (PathwayFamily::Cosine, &s["cosine_".len()..]),
        s if s.starts_with("modified_entropy_") => (
            PathwayFamily::ModifiedEntropy,
            &s["modified_entropy_".len()..],
        ),
        s if s.starts_with("entropy_") => (PathwayFamily::Entropy, &s["entropy_".len()..]),
        other => (PathwayFamily::Cosine, other),
    };
    let mut mz = 0.0_f64;
    let mut intensity = 1.0_f64;
    let mut weighted: Option<bool> = None;
    for part in rest.split('_') {
        if let Some(rest_mz) = part.strip_prefix("mz") {
            mz = rest_mz.parse::<f64>().unwrap_or(0.0);
        } else if let Some(rest_int) = part.strip_prefix("int") {
            intensity = rest_int.parse::<f64>().unwrap_or(1.0);
        } else if let Some(rest_weighted) = part.strip_prefix("weighted") {
            weighted = Some(rest_weighted == "true");
        }
    }
    ParsedConfig {
        family,
        mz,
        intensity,
        weighted,
    }
}

/// Build one `PathwayLineSeries` per distinct config in `rows`, with points
/// `(peak_count, metric_value)` filtered to finite values.
fn build_series_from_aggregate(
    rows: &[DiscriminabilityRow],
    metric: PathwayMetric,
) -> Vec<PathwayLineSeries> {
    let mut per_config: BTreeMap<String, Vec<(i32, f64)>> = BTreeMap::new();
    for row in rows {
        // The CLI plot loop only ever invokes this with AUROC or AUPRC,
        // since `DiscriminabilityRow` carries no accuracy / MCC. The
        // exhaustive match keeps clippy happy and surfaces a clear panic
        // if a future call site forgets that.
        let value = match metric {
            PathwayMetric::Auroc => row.auroc,
            PathwayMetric::Auprc => row.auprc,
            PathwayMetric::Accuracy | PathwayMetric::Mcc => {
                unreachable!("CLI plot loop only emits AUROC / AUPRC, never accuracy or MCC")
            }
        };
        if !value.is_finite() {
            continue;
        }
        let Ok(peak) = i32::try_from(row.peak_count) else {
            continue;
        };
        per_config
            .entry(row.config.clone())
            .or_default()
            .push((peak, value));
    }
    per_config
        .into_iter()
        .map(|(config, points)| {
            let parsed = parse_config_slug(&config);
            PathwayLineSeries {
                label: pretty_series_label(
                    parsed.family,
                    parsed.mz,
                    parsed.intensity,
                    parsed.weighted,
                ),
                family: parsed.family,
                mz_exp: parsed.mz,
                intensity_exp: parsed.intensity,
                weighted: parsed.weighted,
                points,
            }
        })
        .collect()
}

/// Read every row of `pathway_discriminability.parquet`. The file is small
/// (tens of kilobytes), so a single pass keeps everything in memory.
fn read_aggregate_rows(path: &Path) -> Result<Vec<DiscriminabilityRow>> {
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading metadata from {}", path.display()))?
        .build()
        .with_context(|| format!("building reader for {}", path.display()))?;
    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch.with_context(|| format!("decoding batch in {}", path.display()))?;
        observe_aggregate_batch(&batch, &mut rows)?;
    }
    Ok(rows)
}

/// Read every row of `pathway_discriminability_per_class.parquet`. The file
/// is at most a few megabytes, so a single pass keeps everything in memory.
fn read_per_class_rows(path: &Path) -> Result<Vec<PerClassPlotRow>> {
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading metadata from {}", path.display()))?
        .build()
        .with_context(|| format!("building reader for {}", path.display()))?;
    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch.with_context(|| format!("decoding batch in {}", path.display()))?;
        observe_per_class_batch(&batch, &mut rows)?;
    }
    Ok(rows)
}

/// Extract `(config, peak_count, auroc, auprc)` from one record batch. Rows
/// where either AUROC or AUPRC is null or non-finite are dropped.
fn observe_aggregate_batch(batch: &RecordBatch, rows: &mut Vec<DiscriminabilityRow>) -> Result<()> {
    let configs = required_column::<StringArray>(batch, "config")?;
    let peak_counts = required_column::<UInt64Array>(batch, "peak_count")?;
    let auroc = required_column::<Float64Array>(batch, "auroc")?;
    let auprc = required_column::<Float64Array>(batch, "auprc")?;
    for row in 0..batch.num_rows() {
        if auroc.is_null(row) || auprc.is_null(row) {
            continue;
        }
        let auroc_value = auroc.value(row);
        let auprc_value = auprc.value(row);
        if !auroc_value.is_finite() || !auprc_value.is_finite() {
            continue;
        }
        rows.push(DiscriminabilityRow {
            config: configs.value(row).to_string(),
            peak_count: peak_counts.value(row),
            auroc: auroc_value,
            auprc: auprc_value,
        });
    }
    Ok(())
}

/// Extract `(config, peak_count, pathway, auroc, auprc)` from one record
/// batch of the per-class parquet. Rows where either AUROC or AUPRC is null
/// or non-finite are dropped.
fn observe_per_class_batch(batch: &RecordBatch, rows: &mut Vec<PerClassPlotRow>) -> Result<()> {
    let configs = required_column::<StringArray>(batch, "config")?;
    let peak_counts = required_column::<UInt64Array>(batch, "peak_count")?;
    let pathways = required_column::<StringArray>(batch, "pathway")?;
    let auroc = required_column::<Float64Array>(batch, "auroc")?;
    let auprc = required_column::<Float64Array>(batch, "auprc")?;
    for row in 0..batch.num_rows() {
        if auroc.is_null(row) || auprc.is_null(row) {
            continue;
        }
        let auroc_value = auroc.value(row);
        let auprc_value = auprc.value(row);
        if !auroc_value.is_finite() || !auprc_value.is_finite() {
            continue;
        }
        rows.push(PerClassPlotRow {
            config: configs.value(row).to_string(),
            peak_count: peak_counts.value(row),
            pathway: pathways.value(row).to_string(),
            auroc: auroc_value,
            auprc: auprc_value,
        });
    }
    Ok(())
}

/// Downcast a parquet column to the expected Arrow array type.
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

/// Write one metric's plot as SVG.
fn write_plot_svg(
    path: &Path,
    metric: PathwayMetric,
    caption: &str,
    series: &[PathwayLineSeries],
) -> Result<()> {
    let root = SVGBackend::new(path, (LINE_PLOT_WIDTH, LINE_PLOT_HEIGHT)).into_drawing_area();
    draw_pathway_lines(&root, caption, metric, series)
        .with_context(|| format!("writing SVG discriminability plot {}", path.display()))
}

/// Write one metric's plot as PNG.
fn write_plot_png(
    path: &Path,
    metric: PathwayMetric,
    caption: &str,
    series: &[PathwayLineSeries],
) -> Result<()> {
    let root = BitMapBackend::new(path, (LINE_PLOT_WIDTH, LINE_PLOT_HEIGHT)).into_drawing_area();
    draw_pathway_lines(&root, caption, metric, series)
        .with_context(|| format!("writing PNG discriminability plot {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_config_slug_handles_cosine() {
        let parsed = parse_config_slug("cosine_mz0.000_int1.000");
        assert!(matches!(parsed.family, PathwayFamily::Cosine));
        assert!((parsed.mz - 0.0).abs() < 1e-9);
        assert!((parsed.intensity - 1.0).abs() < 1e-9);
        assert!(parsed.weighted.is_none());
    }

    #[test]
    fn parse_config_slug_handles_modified_cosine() {
        let parsed = parse_config_slug("modified_cosine_mz3.000_int0.600");
        assert!(matches!(parsed.family, PathwayFamily::ModifiedCosine));
        assert!((parsed.mz - 3.0).abs() < 1e-9);
        assert!((parsed.intensity - 0.6).abs() < 1e-9);
    }

    #[test]
    fn parse_config_slug_handles_entropy_weighted_true() {
        let parsed = parse_config_slug("entropy_mz0.000_int1.000_weightedtrue");
        assert!(matches!(parsed.family, PathwayFamily::Entropy));
        assert_eq!(parsed.weighted, Some(true));
    }

    #[test]
    fn parse_config_slug_handles_modified_entropy_weighted_false() {
        let parsed = parse_config_slug("modified_entropy_mz0.000_int1.000_weightedfalse");
        assert!(matches!(parsed.family, PathwayFamily::ModifiedEntropy));
        assert_eq!(parsed.weighted, Some(false));
    }
}

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
//! - `auroc.{svg,png}` and `auprc.{svg,png}` at the subdirectory root —
//!   the pooled (micro-averaged) classifier across all queries.
//! - `per_class/<pathway_slug>/{auroc,auprc}.{svg,png}` — the per-pathway
//!   one-vs-rest curves, one chart family per pathway.
//!
//! Each chart draws one line per similarity config: colour encodes the
//! metric family (cosine / modified-cosine / entropy / modified-entropy),
//! dash pattern encodes the m/z exponent (0.0 → solid, 1.0 → dashed,
//! 3.0 → dotted), and a colour mix factor distinguishes the intensity
//! exponent or entropy weighting within a family + dash combination.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, Result};
use arrow_array::{Array, Float64Array, RecordBatch, StringArray, UInt64Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use plotters::{
    coord::Shift,
    element::DashedPathElement,
    prelude::{
        BLACK, BitMapBackend, ChartBuilder, DashedLineSeries, DrawingArea, DrawingBackend,
        IntoDrawingArea, LineSeries, PathElement, SVGBackend, SeriesLabelPosition, WHITE,
    },
    style::{Color, IntoFont, RGBColor, ShapeStyle},
};

use crate::{
    pathway_artifacts::{LINE_PLOT_HEIGHT, LINE_PLOT_WIDTH, usize_to_i32},
    progress::ScanProgress,
    visualize::{ensure_heatmap_font, plotters_error, sanitize_path_component},
};

/// Subdirectory holding the rendered discriminability plots.
const PLOTS_SUBDIR: &str = "pathway_discriminability_plots";
/// Stroke width used for every line in the discriminability plots.
const LINE_STROKE_WIDTH: u32 = 2;
/// Cosine family colour.
const COLOR_COSINE: RGBColor = RGBColor(0x1f, 0x77, 0xb4);
/// Modified cosine family colour.
const COLOR_MODIFIED_COSINE: RGBColor = RGBColor(0xff, 0x7f, 0x0e);
/// Entropy family colour.
const COLOR_ENTROPY: RGBColor = RGBColor(0x2c, 0xa0, 0x2c);
/// Modified entropy family colour.
const COLOR_MODIFIED_ENTROPY: RGBColor = RGBColor(0xd6, 0x27, 0x28);

/// Read `pathway_discriminability.parquet` and
/// `pathway_discriminability_per_class.parquet` and write per-metric line
/// plots: a micro-averaged aggregate at the root of the plots directory
/// plus one one-vs-rest chart pair per pathway under `per_class/`.
///
/// Silently returns `Ok(())` if both parquets are missing or empty —
/// these states correspond to datasets without pathway annotations (e.g.
/// gems-sampled) and should not abort the downstream pipeline.
pub fn write_pathway_discriminability_plots(
    output_dir: &Path,
    progress: &ScanProgress,
) -> Result<()> {
    ensure_heatmap_font()?;
    let plots_dir = output_dir.join(PLOTS_SUBDIR);
    let mut produced_any = false;

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
            for metric in [DiscriminabilityMetric::Auroc, DiscriminabilityMetric::Auprc] {
                let caption = format!("{} (micro-averaged)", metric.title());
                let stem = plots_dir.join(metric.file_stem());
                write_plot_svg(&stem.with_extension("svg"), metric, &caption, &rows)?;
                write_plot_png(&stem.with_extension("png"), metric, &caption, &rows)?;
            }
            render_progress.finish();
            produced_any = true;
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
                for metric in [DiscriminabilityMetric::Auroc, DiscriminabilityMetric::Auprc] {
                    let caption =
                        format!("{} — one-vs-rest, pathway = {}", metric.title(), pathway);
                    let stem = subdir.join(metric.file_stem());
                    write_plot_svg(&stem.with_extension("svg"), metric, &caption, &rows)?;
                    write_plot_png(&stem.with_extension("png"), metric, &caption, &rows)?;
                }
                render_progress.inc(1);
            }
            render_progress.finish();
            produced_any = true;
        }
    }

    let _ = produced_any;
    Ok(())
}

/// Group per-class rows by pathway, returning an ordered map keyed by
/// pathway label so per-pathway plot directories appear in stable order.
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

/// One row of `pathway_discriminability.parquet` reduced to the columns
/// the renderer cares about.
struct DiscriminabilityRow {
    /// Similarity-config slug, e.g. `modified_cosine_mz0.000_int1.000`.
    config: String,
    /// Number of retained top-intensity peaks for this row.
    peak_count: u64,
    /// Area under the ROC curve for the pathway-pair classifier.
    auroc: f64,
    /// Area under the precision-recall curve for the pathway-pair classifier.
    auprc: f64,
}

/// One row of `pathway_discriminability_per_class.parquet` reduced to the
/// columns the renderer cares about. `pathway` is the fixed positive
/// class for the one-vs-rest classifier.
struct PerClassPlotRow {
    /// Similarity-config slug.
    config: String,
    /// Number of retained top-intensity peaks for this row.
    peak_count: u64,
    /// Positive-class pathway label for this row.
    pathway: String,
    /// Area under the ROC curve for the one-vs-rest classifier.
    auroc: f64,
    /// Area under the precision-recall curve for the one-vs-rest classifier.
    auprc: f64,
}

/// Metric rendered as a discriminability line plot.
#[derive(Clone, Copy)]
enum DiscriminabilityMetric {
    /// Area under the ROC curve.
    Auroc,
    /// Area under the precision-recall curve.
    Auprc,
}

impl DiscriminabilityMetric {
    /// Artifact file stem (no extension).
    const fn file_stem(self) -> &'static str {
        match self {
            Self::Auroc => "auroc",
            Self::Auprc => "auprc",
        }
    }

    /// Plot title fragment.
    const fn title(self) -> &'static str {
        match self {
            Self::Auroc => "Pathway-pair AUROC",
            Self::Auprc => "Pathway-pair AUPRC",
        }
    }

    /// Y-axis label.
    const fn y_label(self) -> &'static str {
        match self {
            Self::Auroc => "AUROC",
            Self::Auprc => "AUPRC",
        }
    }

    /// Read the metric value from a row.
    const fn value(self, row: &DiscriminabilityRow) -> f64 {
        match self {
            Self::Auroc => row.auroc,
            Self::Auprc => row.auprc,
        }
    }
}

/// Compute a focused y-axis range for one metric across `rows`, padding the
/// observed min/max by 8 % of the span so curves don't touch the chart
/// frame. Clipped to `[0.0, 1.0]` since AUROC and AUPRC are bounded.
fn focused_y_range(metric: DiscriminabilityMetric, rows: &[DiscriminabilityRow]) -> (f64, f64) {
    let (min, max) = rows
        .iter()
        .map(|row| metric.value(row))
        .filter(|value| value.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), value| {
            (lo.min(value), hi.max(value))
        });
    if !min.is_finite() || !max.is_finite() {
        return (0.0, 1.0);
    }
    let span = (max - min).max(0.01);
    let pad = span * 0.08;
    let lower = (min - pad).max(0.0);
    let upper = (max + pad).min(1.0);
    if upper - lower < 0.01 {
        ((lower - 0.005).max(0.0), (upper + 0.005).min(1.0))
    } else {
        (lower, upper)
    }
}

/// Similarity-metric family parsed from a config slug.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    /// Direct cosine similarity.
    Cosine,
    /// Modified cosine similarity (m/z-aware peak alignment).
    ModifiedCosine,
    /// Direct entropy similarity.
    Entropy,
    /// Modified entropy similarity (m/z-aware peak alignment).
    ModifiedEntropy,
}

impl Family {
    /// Stable rank used to group the legend by family.
    const fn rank(self) -> u8 {
        match self {
            Self::Cosine => 0,
            Self::ModifiedCosine => 1,
            Self::Entropy => 2,
            Self::ModifiedEntropy => 3,
        }
    }

    /// Base colour for the family.
    const fn color(self) -> RGBColor {
        match self {
            Self::Cosine => COLOR_COSINE,
            Self::ModifiedCosine => COLOR_MODIFIED_COSINE,
            Self::Entropy => COLOR_ENTROPY,
            Self::ModifiedEntropy => COLOR_MODIFIED_ENTROPY,
        }
    }
}

/// Decoded config slug.
struct ConfigStyle {
    /// Base similarity family.
    family: Family,
    /// m/z exponent applied during peak weighting.
    mz: f64,
    /// Intensity exponent applied during peak weighting.
    intensity: f64,
    /// Optional entropy-specific peak-weighting flag.
    weighted: Option<bool>,
}

/// Dash pattern (`size`, `spacing`) in pixels. `None` means a solid line.
type DashPattern = Option<(u32, u32)>;

/// Concrete drawing style for one series.
struct SeriesStyle {
    /// Stroke colour and width.
    shape: ShapeStyle,
    /// Optional dash pattern; `None` draws a solid line.
    dash: DashPattern,
}

/// Parse a config slug like `modified_cosine_mz0.000_int1.000` or
/// `entropy_mz0.000_int1.000_weightedtrue` into its components. Unknown
/// slugs fall back to `Family::Cosine` with neutral parameters.
fn parse_config_style(slug: &str) -> ConfigStyle {
    let (family, rest) = match slug {
        s if s.starts_with("modified_cosine_") => {
            (Family::ModifiedCosine, &s["modified_cosine_".len()..])
        }
        s if s.starts_with("cosine_") => (Family::Cosine, &s["cosine_".len()..]),
        s if s.starts_with("modified_entropy_") => {
            (Family::ModifiedEntropy, &s["modified_entropy_".len()..])
        }
        s if s.starts_with("entropy_") => (Family::Entropy, &s["entropy_".len()..]),
        other => (Family::Cosine, other),
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
    ConfigStyle {
        family,
        mz,
        intensity,
        weighted,
    }
}

/// Map a parsed config style to the concrete plotters line style.
fn series_style(config: &ConfigStyle) -> SeriesStyle {
    let dash: DashPattern = if (config.mz - 0.0).abs() < 0.5 {
        None
    } else if (config.mz - 1.0).abs() < 0.5 {
        Some((10, 6))
    } else {
        Some((3, 5))
    };
    let intensity_factor = if (config.intensity - 1.0).abs() < 0.05 {
        1.0
    } else if config.intensity >= 0.5 {
        0.78
    } else {
        0.5
    };
    let weighted_factor = match config.weighted {
        Some(false) => 0.62,
        _ => 1.0,
    };
    let mix = intensity_factor * weighted_factor;
    let base = config.family.color();
    let shape: ShapeStyle = if (mix - 1.0_f64).abs() < 0.01 {
        base.stroke_width(LINE_STROKE_WIDTH)
    } else {
        base.mix(mix).stroke_width(LINE_STROKE_WIDTH)
    };
    SeriesStyle { shape, dash }
}

/// Read every row of `pathway_discriminability.parquet` into memory.
/// The file is at most a few hundred kilobytes, so a single pass is fine.
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

/// Read every row of `pathway_discriminability_per_class.parquet` into
/// memory. The file is at most a few megabytes, so a single pass is fine.
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

/// Extract `(config, peak_count, auroc, auprc)` from one record batch.
/// Rows where either AUROC or AUPRC is null or non-finite are dropped.
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
/// batch of the per-class parquet. Rows where either AUROC or AUPRC is
/// null or non-finite are dropped.
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
    metric: DiscriminabilityMetric,
    caption: &str,
    rows: &[DiscriminabilityRow],
) -> Result<()> {
    let root = SVGBackend::new(path, (LINE_PLOT_WIDTH, LINE_PLOT_HEIGHT)).into_drawing_area();
    draw_plot(&root, metric, caption, rows)
        .with_context(|| format!("writing SVG discriminability plot {}", path.display()))
}

/// Write one metric's plot as PNG.
fn write_plot_png(
    path: &Path,
    metric: DiscriminabilityMetric,
    caption: &str,
    rows: &[DiscriminabilityRow],
) -> Result<()> {
    let root = BitMapBackend::new(path, (LINE_PLOT_WIDTH, LINE_PLOT_HEIGHT)).into_drawing_area();
    draw_plot(&root, metric, caption, rows)
        .with_context(|| format!("writing PNG discriminability plot {}", path.display()))
}

/// Draw one metric's line plot with one curve per config.
fn draw_plot<Backend>(
    root: &DrawingArea<Backend, Shift>,
    metric: DiscriminabilityMetric,
    caption: &str,
    rows: &[DiscriminabilityRow],
) -> Result<()>
where
    Backend: DrawingBackend,
    Backend::ErrorType: std::fmt::Debug,
{
    root.fill(&WHITE).map_err(plotters_error)?;
    let (y_min, y_max) = focused_y_range(metric, rows);
    let x_end = i32::try_from(largest_peak_count(rows).saturating_add(1))
        .context("peak count axis upper bound does not fit i32")?;
    let mut chart = ChartBuilder::on(root)
        .caption(caption, ("sans-serif", 26))
        .margin(24)
        .x_label_area_size(52)
        .y_label_area_size(68)
        .right_y_label_area_size(0)
        .build_cartesian_2d(1_i32..x_end, y_min..y_max)
        .map_err(plotters_error)?;
    chart
        .configure_mesh()
        .x_desc("Retained peak count")
        .y_desc(metric.y_label())
        .x_labels(8)
        .y_labels(8)
        .axis_desc_style(("sans-serif", 20))
        .label_style(("sans-serif", 15))
        .draw()
        .map_err(plotters_error)?;

    let per_config = group_by_config(rows);
    let sorted = sort_configs_for_legend(&per_config);

    for (config, style, config_rows) in sorted {
        let mut points: Vec<(i32, f64)> = config_rows
            .iter()
            .filter_map(|row| {
                let value = metric.value(row);
                if !value.is_finite() {
                    return None;
                }
                Some((usize_to_i32(row.peak_count as usize).ok()?, value))
            })
            .collect();
        points.sort_by_key(|&(x, _)| x);
        if points.is_empty() {
            continue;
        }
        let label = config.clone();
        let series_style = style.shape;
        match style.dash {
            None => {
                chart
                    .draw_series(LineSeries::new(points, series_style))
                    .map_err(plotters_error)?
                    .label(label)
                    .legend(move |(x, y)| {
                        PathElement::new(vec![(x, y), (x + 28, y)], series_style)
                    });
            }
            Some((size, spacing)) => {
                chart
                    .draw_series(DashedLineSeries::new(points, size, spacing, series_style))
                    .map_err(plotters_error)?
                    .label(label)
                    .legend(move |(x, y)| {
                        // Mirror the chart's dash pattern in the legend
                        // swatch so dashed (mz=1.0) and dotted (mz=3.0)
                        // series read as visibly different from the
                        // solid (mz=0.0) ones. `PathElement` with stacked
                        // segments would render as a single polyline
                        // (the per-segment gap is filled in), so we have
                        // to use the same `DashedPathElement` that the
                        // series itself emits.
                        DashedPathElement::new(
                            vec![(x, y), (x + 28, y)],
                            size,
                            spacing,
                            series_style,
                        )
                    });
            }
        }
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .background_style(WHITE.mix(0.82))
        .border_style(BLACK)
        .label_font(("sans-serif", 12).into_font())
        .draw()
        .map_err(plotters_error)?;
    root.present().map_err(plotters_error)
}

/// Largest peak count seen in `rows`, defaulting to the
/// pathway-shard peak grid upper bound when no rows are present.
fn largest_peak_count(rows: &[DiscriminabilityRow]) -> u64 {
    rows.iter().map(|row| row.peak_count).max().unwrap_or(128)
}

/// Group rows by config name and return an ordered map keyed by config slug.
fn group_by_config(rows: &[DiscriminabilityRow]) -> BTreeMap<String, Vec<&DiscriminabilityRow>> {
    let mut per_config: BTreeMap<String, Vec<&DiscriminabilityRow>> = BTreeMap::new();
    for row in rows {
        per_config.entry(row.config.clone()).or_default().push(row);
    }
    per_config
}

/// Sort configs by (family, mz, intensity desc, weighted desc) and return
/// per-config series styles paired with their unsorted row references; the
/// caller turns those rows into a metric-specific point sequence.
fn sort_configs_for_legend<'rows>(
    per_config: &BTreeMap<String, Vec<&'rows DiscriminabilityRow>>,
) -> Vec<(String, SeriesStyle, Vec<&'rows DiscriminabilityRow>)> {
    let mut entries: Vec<(ConfigStyle, String, Vec<&DiscriminabilityRow>)> = per_config
        .iter()
        .map(|(config, rows)| (parse_config_style(config), config.clone(), rows.clone()))
        .collect();
    entries.sort_by(|a, b| {
        let style_a = &a.0;
        let style_b = &b.0;
        style_a
            .family
            .rank()
            .cmp(&style_b.family.rank())
            .then_with(|| float_cmp(style_a.mz, style_b.mz))
            .then_with(|| float_cmp(style_b.intensity, style_a.intensity))
            .then_with(|| weighted_rank(style_a.weighted).cmp(&weighted_rank(style_b.weighted)))
            .then_with(|| a.1.cmp(&b.1))
    });
    entries
        .into_iter()
        .map(|(style, config, rows)| (config, series_style(&style), rows))
        .collect()
}

/// Compare two `f64`s for sort ordering, treating `NaN` as the smallest value.
fn float_cmp(a: f64, b: f64) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
}

/// Stable ordering for the optional `weighted` flag: `true` before `false`,
/// neither value before either of the booleans.
const fn weighted_rank(weighted: Option<bool>) -> u8 {
    match weighted {
        None => 0,
        Some(true) => 1,
        Some(false) => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_config_style_handles_cosine() {
        let parsed = parse_config_style("cosine_mz0.000_int1.000");
        assert!(matches!(parsed.family, Family::Cosine));
        assert!((parsed.mz - 0.0).abs() < 1e-9);
        assert!((parsed.intensity - 1.0).abs() < 1e-9);
        assert!(parsed.weighted.is_none());
    }

    #[test]
    fn parse_config_style_handles_modified_cosine() {
        let parsed = parse_config_style("modified_cosine_mz3.000_int0.600");
        assert!(matches!(parsed.family, Family::ModifiedCosine));
        assert!((parsed.mz - 3.0).abs() < 1e-9);
        assert!((parsed.intensity - 0.6).abs() < 1e-9);
    }

    #[test]
    fn parse_config_style_handles_entropy_weighted_true() {
        let parsed = parse_config_style("entropy_mz0.000_int1.000_weightedtrue");
        assert!(matches!(parsed.family, Family::Entropy));
        assert_eq!(parsed.weighted, Some(true));
    }

    #[test]
    fn parse_config_style_handles_modified_entropy_weighted_false() {
        let parsed = parse_config_style("modified_entropy_mz0.000_int1.000_weightedfalse");
        assert!(matches!(parsed.family, Family::ModifiedEntropy));
        assert_eq!(parsed.weighted, Some(false));
    }
}

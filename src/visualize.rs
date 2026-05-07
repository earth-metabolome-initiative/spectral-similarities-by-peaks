//! Static heatmap rendering for full peak-count comparison grids.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail};
use colorous::Gradient;
use ndarray::{ArrayView2, ArrayView3, Axis};
use plotters::{
    coord::Shift,
    prelude::{
        BitMapBackend, ChartBuilder, DrawingArea, DrawingBackend, IntoDrawingArea, RGBColor,
        Rectangle, SVGBackend, WHITE,
    },
    style::Color,
};

use crate::{
    output::GridArrays,
    progress::{ProgressTask, ScanProgress},
};

/// Width of each rendered heatmap image in pixels.
const IMAGE_WIDTH: u32 = 1_000;
/// Height of each rendered heatmap image in pixels.
const IMAGE_HEIGHT: u32 = 900;
/// Width reserved for the chart area before the colorbar.
const CHART_AREA_WIDTH: u32 = 860;
/// Number of colored rectangles used to draw the colorbar.
const COLORBAR_STEPS: usize = 256;
/// Fallback color for non-finite matrix entries.
const NON_FINITE_COLOR: RGBColor = RGBColor(180, 180, 180);

/// Number of rendered metrics per similarity configuration.
const HEATMAP_METRIC_COUNT: usize = 4;

/// Write SVG and PNG heatmaps for each dense grid matrix and config.
pub fn write_heatmaps(
    output_dir: &Path,
    configs: &[String],
    arrays: &GridArrays,
    progress: &ScanProgress,
) -> Result<()> {
    validate_config_axis(configs, arrays)?;
    let output_dir = output_dir.join("heatmaps");
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("creating {}", output_dir.display()))?;

    let scales = HeatmapScales::from_arrays(arrays);
    let total_heatmaps = configs
        .len()
        .checked_mul(HEATMAP_METRIC_COUNT)
        .and_then(|count| count.checked_mul(2))
        .unwrap_or(usize::MAX);
    let task = progress.bar(
        u64::try_from(total_heatmaps).unwrap_or(u64::MAX),
        "rendering heatmaps",
    );
    for (config_index, config) in configs.iter().enumerate() {
        let config_dir = output_dir.join(sanitize_path_component(config));
        fs::create_dir_all(&config_dir)
            .with_context(|| format!("creating {}", config_dir.display()))?;
        for metric in heatmap_metrics(arrays, &scales, config_index) {
            task.set_message(format!("rendering {config} {}", metric.name));
            write_heatmap_pair(&config_dir, config, &metric, &task)?;
        }
    }
    task.finish();
    Ok(())
}

/// Validate that all dense matrices share the same config axis.
fn validate_config_axis(configs: &[String], arrays: &GridArrays) -> Result<()> {
    for (name, axis_len) in [
        ("mean_delta", arrays.mean_delta.shape()[0]),
        ("ks_statistic", arrays.ks_statistic.shape()[0]),
        (
            "ks_pvalue_asymptotic",
            arrays.ks_pvalue_asymptotic.shape()[0],
        ),
        ("wasserstein_1d", arrays.wasserstein_1d.shape()[0]),
    ] {
        if axis_len != configs.len() {
            bail!(
                "{name} config axis has length {axis_len}, expected {}",
                configs.len()
            );
        }
    }
    Ok(())
}

/// Return metric views for one similarity config.
fn heatmap_metrics<'a>(
    arrays: &'a GridArrays,
    scales: &'a HeatmapScales,
    config_index: usize,
) -> [HeatmapMetric<'a>; 4] {
    [
        HeatmapMetric {
            name: "mean_delta",
            title: "Mean delta",
            values: arrays.mean_delta.index_axis(Axis(0), config_index),
            scale: scales.mean_delta,
            palette: colorous::RED_BLUE,
        },
        HeatmapMetric {
            name: "ks_statistic",
            title: "KS statistic",
            values: arrays.ks_statistic.index_axis(Axis(0), config_index),
            scale: scales.unit_interval,
            palette: colorous::VIRIDIS,
        },
        HeatmapMetric {
            name: "ks_pvalue_asymptotic",
            title: "Asymptotic KS p-value",
            values: arrays
                .ks_pvalue_asymptotic
                .index_axis(Axis(0), config_index),
            scale: scales.unit_interval,
            palette: colorous::VIRIDIS,
        },
        HeatmapMetric {
            name: "wasserstein_1d",
            title: "1D Wasserstein",
            values: arrays.wasserstein_1d.index_axis(Axis(0), config_index),
            scale: scales.wasserstein_1d,
            palette: colorous::VIRIDIS,
        },
    ]
}

/// Write both SVG and PNG files for one heatmap metric.
fn write_heatmap_pair(
    output_dir: &Path,
    config: &str,
    metric: &HeatmapMetric<'_>,
    progress: &ProgressTask,
) -> Result<()> {
    let stem = output_dir.join(metric.name);
    write_svg(&stem.with_extension("svg"), config, metric)?;
    progress.inc(1);
    write_png(&stem.with_extension("png"), config, metric)?;
    progress.inc(1);
    Ok(())
}

/// Write one SVG heatmap.
fn write_svg(path: &Path, config: &str, metric: &HeatmapMetric<'_>) -> Result<()> {
    let root = SVGBackend::new(path, (IMAGE_WIDTH, IMAGE_HEIGHT)).into_drawing_area();
    draw_heatmap(&root, config, metric)
        .with_context(|| format!("writing SVG heatmap {}", path.display()))
}

/// Write one PNG heatmap.
fn write_png(path: &Path, config: &str, metric: &HeatmapMetric<'_>) -> Result<()> {
    let root = BitMapBackend::new(path, (IMAGE_WIDTH, IMAGE_HEIGHT)).into_drawing_area();
    draw_heatmap(&root, config, metric)
        .with_context(|| format!("writing PNG heatmap {}", path.display()))
}

/// Draw one heatmap into a concrete backend.
fn draw_heatmap<Backend>(
    root: &DrawingArea<Backend, Shift>,
    config: &str,
    metric: &HeatmapMetric<'_>,
) -> Result<()>
where
    Backend: DrawingBackend,
    Backend::ErrorType: std::fmt::Debug,
{
    root.fill(&WHITE).map_err(plotters_error)?;
    let (chart_area, colorbar_area) = root.split_horizontally(CHART_AREA_WIDTH);
    draw_matrix(&chart_area, config, metric)?;
    draw_colorbar(&colorbar_area, metric)?;
    root.present().map_err(plotters_error)
}

/// Draw the main matrix panel.
fn draw_matrix<Backend>(
    area: &DrawingArea<Backend, Shift>,
    config: &str,
    metric: &HeatmapMetric<'_>,
) -> Result<()>
where
    Backend: DrawingBackend,
    Backend::ErrorType: std::fmt::Debug,
{
    let rows = metric.values.nrows();
    let columns = metric.values.ncols();
    let x_end = usize_to_i32(columns + 1)?;
    let y_end = usize_to_i32(rows + 1)?;
    let mut chart = ChartBuilder::on(area)
        .caption(format!("{config} / {}", metric.title), ("sans-serif", 24))
        .margin(22)
        .x_label_area_size(48)
        .y_label_area_size(58)
        .build_cartesian_2d(1_i32..x_end, 1_i32..y_end)
        .map_err(plotters_error)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("Peak count B")
        .y_desc("Peak count A")
        .x_labels(5)
        .y_labels(5)
        .axis_desc_style(("sans-serif", 20))
        .label_style(("sans-serif", 16))
        .draw()
        .map_err(plotters_error)?;

    chart
        .draw_series(matrix_cells(metric)?)
        .map_err(plotters_error)?;
    Ok(())
}

/// Draw the metric colorbar panel.
fn draw_colorbar<Backend>(
    area: &DrawingArea<Backend, Shift>,
    metric: &HeatmapMetric<'_>,
) -> Result<()>
where
    Backend: DrawingBackend,
    Backend::ErrorType: std::fmt::Debug,
{
    let (minimum, maximum) = metric.scale.range();
    let mut chart = ChartBuilder::on(area)
        .caption(metric.name, ("sans-serif", 16))
        .margin_left(8)
        .margin_right(24)
        .margin_top(90)
        .margin_bottom(86)
        .build_cartesian_2d(0.0_f64..1.0_f64, minimum..maximum)
        .map_err(plotters_error)?;
    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_labels(0)
        .y_labels(7)
        .label_style(("sans-serif", 15))
        .y_label_formatter(&|value| format_tick(*value))
        .draw()
        .map_err(plotters_error)?;
    chart
        .draw_series(colorbar_cells(metric))
        .map_err(plotters_error)?;
    Ok(())
}

/// Return all colored matrix cells for a heatmap.
fn matrix_cells(metric: &HeatmapMetric<'_>) -> Result<Vec<Rectangle<(i32, i32)>>> {
    let mut cells = Vec::with_capacity(metric.values.len());
    for row in 0..metric.values.nrows() {
        for column in 0..metric.values.ncols() {
            let x0 = usize_to_i32(column + 1)?;
            let y0 = usize_to_i32(row + 1)?;
            cells.push(Rectangle::new(
                [(x0, y0), (x0 + 1, y0 + 1)],
                metric.color(metric.values[[row, column]]).filled(),
            ));
        }
    }
    Ok(cells)
}

/// Return colored cells for the vertical colorbar.
fn colorbar_cells(metric: &HeatmapMetric<'_>) -> Vec<Rectangle<(f64, f64)>> {
    let (minimum, maximum) = metric.scale.range();
    let span = maximum - minimum;
    let mut cells = Vec::with_capacity(COLORBAR_STEPS);
    for step in 0..COLORBAR_STEPS {
        let lower = minimum + span * step as f64 / COLORBAR_STEPS as f64;
        let upper = minimum + span * (step + 1) as f64 / COLORBAR_STEPS as f64;
        let sample = f64::midpoint(lower, upper);
        cells.push(Rectangle::new(
            [(0.0, lower), (1.0, upper)],
            metric.color(sample).filled(),
        ));
    }
    cells
}

/// Metric matrix and rendering parameters.
struct HeatmapMetric<'a> {
    /// Stable file stem for the metric.
    name: &'static str,
    /// Human-readable title for the metric.
    title: &'static str,
    /// Matrix values for one similarity config.
    values: ArrayView2<'a, f64>,
    /// Value scale used to normalize colors.
    scale: HeatmapScale,
    /// Color palette used to render values.
    palette: Gradient,
}

impl HeatmapMetric<'_> {
    /// Return the plotting color for a matrix value.
    fn color(&self, value: f64) -> RGBColor {
        if !value.is_finite() {
            return NON_FINITE_COLOR;
        }
        let color = self.palette.eval_continuous(self.scale.normalize(value));
        RGBColor(color.r, color.g, color.b)
    }
}

/// Shared value scales for all rendered metrics.
struct HeatmapScales {
    /// Symmetric scale for mean-delta heatmaps.
    mean_delta: HeatmapScale,
    /// Fixed `[0, 1]` scale.
    unit_interval: HeatmapScale,
    /// Global nonnegative scale for Wasserstein heatmaps.
    wasserstein_1d: HeatmapScale,
}

impl HeatmapScales {
    /// Build global scales from all dense matrices.
    fn from_arrays(arrays: &GridArrays) -> Self {
        Self {
            mean_delta: HeatmapScale::diverging_zero(max_abs(arrays.mean_delta.view())),
            unit_interval: HeatmapScale::sequential(0.0, 1.0),
            wasserstein_1d: HeatmapScale::sequential(
                0.0,
                finite_max(arrays.wasserstein_1d.view()).unwrap_or(1.0),
            ),
        }
    }
}

/// Value scale used to normalize one metric.
#[derive(Clone, Copy)]
enum HeatmapScale {
    /// Sequential scale from minimum to maximum.
    Sequential {
        /// Lower bound.
        minimum: f64,
        /// Upper bound.
        maximum: f64,
    },
    /// Diverging scale centered at zero.
    DivergingZero {
        /// Absolute bound on both sides of zero.
        maximum_abs: f64,
    },
}

impl HeatmapScale {
    /// Create a sequential scale with a nonzero span.
    fn sequential(minimum: f64, maximum: f64) -> Self {
        let maximum = if maximum.is_finite() && maximum > minimum {
            maximum
        } else {
            minimum + 1.0
        };
        Self::Sequential { minimum, maximum }
    }

    /// Create a zero-centered diverging scale.
    fn diverging_zero(maximum_abs: f64) -> Self {
        let maximum_abs = if maximum_abs.is_finite() && maximum_abs > 0.0 {
            maximum_abs
        } else {
            1.0
        };
        Self::DivergingZero { maximum_abs }
    }

    /// Return the displayed numeric range.
    const fn range(self) -> (f64, f64) {
        match self {
            Self::Sequential { minimum, maximum } => (minimum, maximum),
            Self::DivergingZero { maximum_abs } => (-maximum_abs, maximum_abs),
        }
    }

    /// Normalize a value to a `[0, 1]` palette coordinate.
    fn normalize(self, value: f64) -> f64 {
        let (minimum, maximum) = self.range();
        ((value - minimum) / (maximum - minimum)).clamp(0.0, 1.0)
    }
}

/// Return the maximum absolute finite value in a matrix.
fn max_abs(values: ArrayView3<'_, f64>) -> f64 {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(f64::abs)
        .fold(0.0, f64::max)
}

/// Return the maximum finite value in a matrix.
fn finite_max(values: ArrayView3<'_, f64>) -> Option<f64> {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .reduce(f64::max)
}

/// Format a colorbar tick value.
fn format_tick(value: f64) -> String {
    if value != 0.0 && !(0.001..1_000.0).contains(&value.abs()) {
        format!("{value:.2e}")
    } else {
        format!("{value:.3}")
    }
}

/// Convert a `usize` to `i32` for plotting coordinates.
fn usize_to_i32(value: usize) -> Result<i32> {
    i32::try_from(value).context("plot coordinate does not fit i32")
}

/// Return a filesystem-safe path component.
fn sanitize_path_component(raw: &str) -> PathBuf {
    let mut sanitized = String::with_capacity(raw.len());
    for character in raw.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
            sanitized.push(character);
        } else {
            sanitized.push('_');
        }
    }
    let sanitized = sanitized.trim_matches('_');
    if sanitized.is_empty() {
        PathBuf::from("config")
    } else {
        PathBuf::from(sanitized)
    }
}

/// Convert a plotters backend error into an anyhow error.
fn plotters_error<Error: std::fmt::Debug>(error: Error) -> anyhow::Error {
    anyhow!("{error:?}")
}

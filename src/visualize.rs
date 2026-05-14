//! Static heatmap rendering for full peak-count comparison grids.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::{
    env, fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use anyhow::{Context, Result, anyhow, bail};
use colorous::Gradient;
use ndarray::{ArrayView2, ArrayView3, Axis};
use plotters::{
    coord::Shift,
    prelude::{
        BLACK, BitMapBackend, ChartBuilder, DrawingArea, DrawingBackend, IntoDrawingArea, RGBColor,
        Rectangle, SVGBackend, WHITE,
    },
    style::{
        Color, FontStyle, IntoFont, register_font,
        text_anchor::{HPos, Pos, VPos},
    },
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
/// Lower bound used when logarithmic scales encounter subnormal values.
const LOG_MINIMUM_POSITIVE: f64 = f64::MIN_POSITIVE;
/// Linear neighborhood around zero for signed logarithmic diverging scales.
const SIGNED_LOG_LINEAR_FRACTION: f64 = 1.0e-3;

/// Number of rendered metrics per similarity configuration.
const HEATMAP_METRIC_COUNT: usize = 8;

/// Horizontal padding between the right edge of the colorbar panel and the
/// right-anchored colorbar label.
const COLORBAR_LABEL_RIGHT_PAD: i32 = 16;
/// Vertical position (pixels from the top of the colorbar panel) for the
/// colorbar label's vertical center. Sits just above the colorbar top edge,
/// which is at `margin_top = 90` pixels of the same panel.
const COLORBAR_LABEL_TOP: i32 = 70;

/// Environment variable that may point to a TrueType or OpenType font file.
const HEATMAP_FONT_ENV_VAR: &str = "SPECTRAL_SIMILARITIES_FONT";

/// Common Linux sans-serif fonts used when no explicit font path is configured.
const HEATMAP_FONT_CANDIDATES: &[&str] = &[
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSans-Regular.ttf",
];

/// Cached result of registering the Plotters sans-serif font.
static HEATMAP_FONT_REGISTRATION: OnceLock<std::result::Result<(), String>> = OnceLock::new();

/// Write SVG and PNG heatmaps for each dense grid matrix and config.
pub fn write_heatmaps(
    output_dir: &Path,
    configs: &[String],
    arrays: &GridArrays,
    progress: &ScanProgress,
) -> Result<()> {
    validate_config_axis(configs, arrays)?;
    ensure_heatmap_font()?;
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

/// Ensure Plotters can render text without native fontconfig or `FreeType` bindings.
pub fn ensure_heatmap_font() -> Result<()> {
    match HEATMAP_FONT_REGISTRATION.get_or_init(register_heatmap_font) {
        Ok(()) => Ok(()),
        Err(message) => bail!("{message}"),
    }
}

/// Register the configured or first available system sans-serif font.
fn register_heatmap_font() -> std::result::Result<(), String> {
    if let Some(path) = env::var_os(HEATMAP_FONT_ENV_VAR) {
        let path = PathBuf::from(path);
        return register_heatmap_font_path(&path).map_err(|error| {
            format!(
                "failed to register font from {HEATMAP_FONT_ENV_VAR}={}: {error}",
                path.display()
            )
        });
    }

    for path in HEATMAP_FONT_CANDIDATES.iter().map(Path::new) {
        match fs::read(path) {
            Ok(bytes) => return register_heatmap_font_bytes(path, bytes),
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => return Err(format!("failed to read {}: {error}", path.display())),
        }
    }

    Err(format!(
        "no usable sans-serif font found; install fonts-dejavu-core or set {HEATMAP_FONT_ENV_VAR}"
    ))
}

/// Read and register one font file as Plotters' `sans-serif` family.
fn register_heatmap_font_path(path: &Path) -> std::result::Result<(), String> {
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    register_heatmap_font_bytes(path, bytes)
}

/// Register in-memory font bytes as Plotters' `sans-serif` family.
fn register_heatmap_font_bytes(path: &Path, bytes: Vec<u8>) -> std::result::Result<(), String> {
    let leaked = Box::leak(bytes.into_boxed_slice());
    register_font("sans-serif", FontStyle::Normal, leaked)
        .map_err(|_| format!("{} is not a valid TrueType/OpenType font", path.display()))
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
///
/// Each underlying metric is emitted twice — once with a linear value scale
/// and once with a (signed-)logarithmic scale — so downstream artifacts always
/// include both views side-by-side.
fn heatmap_metrics<'a>(
    arrays: &'a GridArrays,
    scales: &'a HeatmapScales,
    config_index: usize,
) -> [HeatmapMetric<'a>; 8] {
    let mean_delta = arrays.mean_delta.index_axis(Axis(0), config_index);
    let ks_statistic = arrays.ks_statistic.index_axis(Axis(0), config_index);
    let ks_pvalue_asymptotic = arrays
        .ks_pvalue_asymptotic
        .index_axis(Axis(0), config_index);
    let wasserstein_1d = arrays.wasserstein_1d.index_axis(Axis(0), config_index);
    [
        HeatmapMetric {
            name: "mean_delta_linear",
            title: "Δ mean (linear)",
            colorbar_label: "mean delta",
            values: mean_delta,
            scale: scales.mean_delta_linear,
            palette: colorous::RED_BLUE,
            diagonal_value: 0.0,
        },
        HeatmapMetric {
            name: "mean_delta_log",
            title: "Δ mean (signed log)",
            colorbar_label: "mean delta",
            values: mean_delta,
            scale: scales.mean_delta_log,
            palette: colorous::RED_BLUE,
            diagonal_value: 0.0,
        },
        HeatmapMetric {
            name: "ks_statistic_linear",
            title: "KS statistic (linear)",
            colorbar_label: "KS",
            values: ks_statistic,
            scale: scales.ks_statistic_linear,
            palette: colorous::VIRIDIS,
            diagonal_value: 0.0,
        },
        HeatmapMetric {
            name: "ks_statistic_log",
            title: "KS statistic (log)",
            colorbar_label: "KS",
            values: ks_statistic,
            scale: scales.ks_statistic_log,
            palette: colorous::VIRIDIS,
            diagonal_value: 0.0,
        },
        HeatmapMetric {
            name: "ks_pvalue_asymptotic_linear",
            title: "KS p-value (linear)",
            colorbar_label: "p-value",
            values: ks_pvalue_asymptotic,
            scale: scales.ks_pvalue_asymptotic_linear,
            palette: colorous::VIRIDIS,
            diagonal_value: 1.0,
        },
        HeatmapMetric {
            name: "ks_pvalue_asymptotic_log",
            title: "KS p-value (log)",
            colorbar_label: "p-value",
            values: ks_pvalue_asymptotic,
            scale: scales.ks_pvalue_asymptotic_log,
            palette: colorous::VIRIDIS,
            diagonal_value: 1.0,
        },
        HeatmapMetric {
            name: "wasserstein_1d_linear",
            title: "Wasserstein (linear)",
            colorbar_label: "Wasserstein",
            values: wasserstein_1d,
            scale: scales.wasserstein_1d_linear,
            palette: colorous::VIRIDIS,
            diagonal_value: 0.0,
        },
        HeatmapMetric {
            name: "wasserstein_1d_log",
            title: "Wasserstein (log)",
            colorbar_label: "Wasserstein",
            values: wasserstein_1d,
            scale: scales.wasserstein_1d_log,
            palette: colorous::VIRIDIS,
            diagonal_value: 0.0,
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
        .caption(
            format!("{} — {}", pretty_config_title(config), metric.title),
            ("sans-serif", 24),
        )
        .margin(22)
        .x_label_area_size(48)
        .y_label_area_size(58)
        .build_cartesian_2d(0_i32..x_end, 0_i32..y_end)
        .map_err(plotters_error)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("Top peaks retained")
        .y_desc("Top peaks retained")
        .x_labels(usize::try_from(x_end).unwrap_or(usize::MAX))
        .y_labels(usize::try_from(y_end).unwrap_or(usize::MAX))
        .x_label_formatter(&|value| {
            if HEATMAP_AXIS_TICKS.contains(value) {
                value.to_string()
            } else {
                String::new()
            }
        })
        .y_label_formatter(&|value| {
            if HEATMAP_AXIS_TICKS.contains(value) {
                value.to_string()
            } else {
                String::new()
            }
        })
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
    // Right-aligned colorbar label drawn manually above the chart.
    // `.caption()` would render centered; manual draw_text gives us the
    // anchoring we want.
    let (area_width, _) = area.dim_in_pixel();
    let label_x = i32::try_from(area_width)?.saturating_sub(COLORBAR_LABEL_RIGHT_PAD);
    let label_style = ("sans-serif", 16)
        .into_font()
        .color(&BLACK)
        .pos(Pos::new(HPos::Right, VPos::Center));
    area.draw_text(
        metric.colorbar_label,
        &label_style,
        (label_x, COLORBAR_LABEL_TOP),
    )
    .map_err(plotters_error)?;

    let mut chart = ChartBuilder::on(area)
        .margin_left(4)
        .margin_right(12)
        .margin_top(90)
        .margin_bottom(86)
        .y_label_area_size(80)
        .build_cartesian_2d(0.0_f64..1.0_f64, 0.0_f64..1.0_f64)
        .map_err(plotters_error)?;
    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_labels(0)
        .y_labels(7)
        .label_style(("sans-serif", 15))
        .y_label_formatter(&|position| format_tick(metric.scale.value_at(*position)))
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
            let value = if row == column {
                metric.diagonal_value
            } else {
                metric.values[[row, column]]
            };
            cells.push(Rectangle::new(
                [(x0, y0), (x0 + 1, y0 + 1)],
                metric.color(value).filled(),
            ));
        }
    }
    Ok(cells)
}

/// Return colored cells for the vertical colorbar.
fn colorbar_cells(metric: &HeatmapMetric<'_>) -> Vec<Rectangle<(f64, f64)>> {
    let mut cells = Vec::with_capacity(COLORBAR_STEPS);
    for step in 0..COLORBAR_STEPS {
        let lower = step as f64 / COLORBAR_STEPS as f64;
        let upper = (step + 1) as f64 / COLORBAR_STEPS as f64;
        let sample = f64::midpoint(lower, upper);
        cells.push(Rectangle::new(
            [(0.0, lower), (1.0, upper)],
            metric.color_at_position(sample).filled(),
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
    /// Short label shown above the colorbar. Avoids the underscore-laden
    /// `metric.name` and uses common scientific shorthand.
    colorbar_label: &'static str,
    /// Matrix values for one similarity config.
    values: ArrayView2<'a, f64>,
    /// Value scale used to normalize colors.
    scale: HeatmapScale,
    /// Color palette used to render values.
    palette: Gradient,
    /// Canonical value to render on the main diagonal regardless of the
    /// stored matrix entry. A distribution compared against itself has
    /// `mean_delta=0`, `ks_statistic=0`, `ks_pvalue_asymptotic=1`, and
    /// `wasserstein_1d=0`; forcing this value here guarantees the diagonal
    /// always shows identity regardless of any upstream computation bug or
    /// pre-fix npz contents.
    diagonal_value: f64,
}

impl HeatmapMetric<'_> {
    /// Return the plotting color for a matrix value.
    fn color(&self, value: f64) -> RGBColor {
        if !value.is_finite() {
            return NON_FINITE_COLOR;
        }
        self.color_at_position(self.scale.normalize(value))
    }

    /// Return the plotting color for an already-normalized palette coordinate.
    fn color_at_position(&self, position: f64) -> RGBColor {
        let color = self.palette.eval_continuous(position.clamp(0.0, 1.0));
        RGBColor(color.r, color.g, color.b)
    }
}

/// Shared value scales for all rendered metrics.
///
/// Every metric carries both a linear and a (signed-)logarithmic scale so we
/// can render the two side-by-side in `heatmap_metrics`.
struct HeatmapScales {
    /// Linear diverging scale for mean-delta heatmaps, centered at zero.
    mean_delta_linear: HeatmapScale,
    /// Signed logarithmic diverging scale for mean-delta heatmaps.
    mean_delta_log: HeatmapScale,
    /// Linear sequential scale for `KS` statistic heatmaps.
    ks_statistic_linear: HeatmapScale,
    /// Global logarithmic scale for `KS` statistic heatmaps.
    ks_statistic_log: HeatmapScale,
    /// Linear sequential scale for asymptotic `KS` p-value heatmaps.
    ks_pvalue_asymptotic_linear: HeatmapScale,
    /// Global logarithmic scale for asymptotic `KS` p-value heatmaps.
    ks_pvalue_asymptotic_log: HeatmapScale,
    /// Linear sequential scale for `Wasserstein` heatmaps.
    wasserstein_1d_linear: HeatmapScale,
    /// Global logarithmic scale for `Wasserstein` heatmaps.
    wasserstein_1d_log: HeatmapScale,
}

impl HeatmapScales {
    /// Build global scales from all dense matrices.
    fn from_arrays(arrays: &GridArrays) -> Self {
        let mean_delta_abs = max_abs(&arrays.mean_delta.view());
        let ks_min = finite_min(&arrays.ks_statistic.view()).unwrap_or(0.0);
        let ks_pos_min = finite_positive_min(&arrays.ks_statistic.view()).unwrap_or(1.0);
        let ks_max = finite_max(&arrays.ks_statistic.view()).unwrap_or(1.0);
        let pv_min = finite_min(&arrays.ks_pvalue_asymptotic.view()).unwrap_or(0.0);
        let pv_pos_min = finite_positive_min(&arrays.ks_pvalue_asymptotic.view()).unwrap_or(1.0);
        let pv_max = finite_max(&arrays.ks_pvalue_asymptotic.view()).unwrap_or(1.0);
        let ws_min = finite_min(&arrays.wasserstein_1d.view()).unwrap_or(0.0);
        let ws_pos_min = finite_positive_min(&arrays.wasserstein_1d.view()).unwrap_or(1.0);
        let ws_max = finite_max(&arrays.wasserstein_1d.view()).unwrap_or(1.0);
        Self {
            mean_delta_linear: HeatmapScale::signed_diverging_zero(mean_delta_abs),
            mean_delta_log: HeatmapScale::signed_log_diverging_zero(mean_delta_abs),
            ks_statistic_linear: HeatmapScale::sequential(ks_min, ks_max),
            ks_statistic_log: HeatmapScale::log_sequential(ks_min, ks_pos_min, ks_max),
            ks_pvalue_asymptotic_linear: HeatmapScale::sequential(pv_min, pv_max),
            ks_pvalue_asymptotic_log: HeatmapScale::log_sequential(pv_min, pv_pos_min, pv_max),
            wasserstein_1d_linear: HeatmapScale::sequential(ws_min, ws_max),
            wasserstein_1d_log: HeatmapScale::log_sequential(ws_min, ws_pos_min, ws_max),
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
    /// Logarithmic sequential scale with a separate finite display minimum.
    LogSequential {
        /// Actual lower bound shown on the colorbar.
        minimum: f64,
        /// Positive floor used for logarithmic normalization.
        positive_floor: f64,
        /// Upper bound.
        maximum: f64,
    },
    /// Signed logarithmic diverging scale centered at zero.
    SignedLogDivergingZero {
        /// Absolute bound on both sides of zero.
        maximum_abs: f64,
        /// Linear neighborhood around zero before the logarithmic region.
        linear_threshold: f64,
    },
    /// Linear diverging scale centered at zero with a symmetric absolute bound.
    SignedDivergingZero {
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

    /// Create a logarithmic sequential scale with a nonzero positive floor.
    fn log_sequential(minimum: f64, positive_minimum: f64, maximum: f64) -> Self {
        let fallback = Self::sequential(minimum, maximum);
        if !maximum.is_finite() || maximum <= 0.0 {
            return fallback;
        }
        let positive_floor = positive_minimum
            .max(LOG_MINIMUM_POSITIVE)
            .min(maximum)
            .max(LOG_MINIMUM_POSITIVE);
        if !positive_floor.is_finite() || positive_floor >= maximum {
            return fallback;
        }
        Self::LogSequential {
            minimum,
            positive_floor,
            maximum,
        }
    }

    /// Create a zero-centered signed logarithmic diverging scale.
    fn signed_log_diverging_zero(maximum_abs: f64) -> Self {
        let maximum_abs = if maximum_abs.is_finite() && maximum_abs > 0.0 {
            maximum_abs
        } else {
            1.0
        };
        let linear_threshold = (maximum_abs * SIGNED_LOG_LINEAR_FRACTION).max(LOG_MINIMUM_POSITIVE);
        Self::SignedLogDivergingZero {
            maximum_abs,
            linear_threshold,
        }
    }

    /// Create a zero-centered linear diverging scale.
    fn signed_diverging_zero(maximum_abs: f64) -> Self {
        let maximum_abs = if maximum_abs.is_finite() && maximum_abs > 0.0 {
            maximum_abs
        } else {
            1.0
        };
        Self::SignedDivergingZero { maximum_abs }
    }

    /// Normalize a value to a `[0, 1]` palette coordinate.
    fn normalize(self, value: f64) -> f64 {
        match self {
            Self::Sequential { minimum, maximum } => {
                ((value - minimum) / (maximum - minimum)).clamp(0.0, 1.0)
            }
            Self::LogSequential {
                positive_floor,
                maximum,
                ..
            } => {
                if value <= positive_floor {
                    return 0.0;
                }
                let log_minimum = positive_floor.ln();
                let log_maximum = maximum.ln();
                ((value.ln() - log_minimum) / (log_maximum - log_minimum)).clamp(0.0, 1.0)
            }
            Self::SignedLogDivergingZero {
                maximum_abs,
                linear_threshold,
            } => {
                let clipped = value.clamp(-maximum_abs, maximum_abs);
                let signed = clipped.signum() * (clipped.abs() / linear_threshold).ln_1p()
                    / (maximum_abs / linear_threshold).ln_1p();
                (0.5_f64).mul_add(signed, 0.5).clamp(0.0, 1.0)
            }
            Self::SignedDivergingZero { maximum_abs } => {
                let clipped = value.clamp(-maximum_abs, maximum_abs);
                ((clipped / (2.0 * maximum_abs)) + 0.5).clamp(0.0, 1.0)
            }
        }
    }

    /// Convert a normalized colorbar coordinate back to the represented value.
    fn value_at(self, position: f64) -> f64 {
        let position = position.clamp(0.0, 1.0);
        match self {
            Self::Sequential { minimum, maximum } => (maximum - minimum).mul_add(position, minimum),
            Self::LogSequential {
                minimum,
                positive_floor,
                maximum,
            } => {
                if position <= f64::EPSILON && minimum <= 0.0 {
                    return minimum;
                }
                let log_minimum = positive_floor.ln();
                let log_maximum = maximum.ln();
                (log_maximum - log_minimum)
                    .mul_add(position, log_minimum)
                    .exp()
            }
            Self::SignedLogDivergingZero {
                maximum_abs,
                linear_threshold,
            } => {
                let signed_position = position.mul_add(2.0, -1.0);
                if signed_position.abs() <= f64::EPSILON {
                    return 0.0;
                }
                let magnitude = linear_threshold
                    * ((maximum_abs / linear_threshold).ln_1p() * signed_position.abs()).exp_m1();
                signed_position.signum() * magnitude
            }
            Self::SignedDivergingZero { maximum_abs } => {
                (2.0 * maximum_abs).mul_add(position, -maximum_abs)
            }
        }
    }
}

/// Return the maximum absolute finite value in a matrix.
fn max_abs(values: &ArrayView3<'_, f64>) -> f64 {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(f64::abs)
        .fold(0.0, f64::max)
}

/// Return the minimum finite value in a matrix.
fn finite_min(values: &ArrayView3<'_, f64>) -> Option<f64> {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .reduce(f64::min)
}

/// Return the maximum finite value in a matrix.
fn finite_max(values: &ArrayView3<'_, f64>) -> Option<f64> {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .reduce(f64::max)
}

/// Return the smallest finite positive value in a matrix.
fn finite_positive_min(values: &ArrayView3<'_, f64>) -> Option<f64> {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .reduce(f64::min)
}

/// Format a colorbar tick value.
///
/// Only an exact `0.0` collapses to the literal `"0"`. Genuinely tiny values
/// (e.g. log-scale p-value ticks like `1e-100`) are not rounded to zero —
/// they flow into scientific notation so the colorbar stays readable across
/// many orders of magnitude.
fn format_tick(value: f64) -> String {
    if value == 0.0 {
        "0".to_string()
    } else if (0.001..1_000.0).contains(&value.abs()) {
        format!("{value:.3}")
    } else {
        format!("{value:.2e}")
    }
}

/// Convert a `usize` to `i32` for plotting coordinates.
fn usize_to_i32(value: usize) -> Result<i32> {
    i32::try_from(value).context("plot coordinate does not fit i32")
}

/// Peak-count axis tick positions kept on every distribution heatmap.
const HEATMAP_AXIS_TICKS: &[i32] = &[0, 32, 64, 96, 128];

/// Pretty-print a `SimilarityConfig::name()` slug as a human-readable title.
///
/// Drops terms whose exponent is 0 (they contribute the identity 1 to the
/// weight product) and omits the `^1` suffix when an exponent equals 1.
/// Examples:
///
/// - `cosine_mz0.000_int1.000` → `Cosine, w ∝ intensity`
/// - `cosine_mz1.000_int0.500` → `Cosine, w ∝ m/z · intensity^0.5`
/// - `cosine_mz3.000_int0.600` → `Cosine, w ∝ m/z^3 · intensity^0.6`
/// - `entropy_mz0.000_int1.000_weightedfalse` → `Unweighted entropy, w ∝ intensity`
fn pretty_config_title(slug: &str) -> String {
    let (family_kind, rest) = if let Some(rest) = slug.strip_prefix("modified_entropy_") {
        (PrettyFamily::ModifiedEntropy, rest)
    } else if let Some(rest) = slug.strip_prefix("entropy_") {
        (PrettyFamily::Entropy, rest)
    } else if let Some(rest) = slug.strip_prefix("modified_cosine_") {
        (PrettyFamily::ModifiedCosine, rest)
    } else if let Some(rest) = slug.strip_prefix("cosine_") {
        (PrettyFamily::Cosine, rest)
    } else {
        return slug.to_string();
    };

    let mut mz_power = 0.0_f64;
    let mut intensity_power = 0.0_f64;
    let mut weighted = None;
    for part in rest.split('_') {
        if let Some(value) = part.strip_prefix("mz") {
            if let Ok(parsed) = value.parse::<f64>() {
                mz_power = parsed;
            }
        } else if let Some(value) = part.strip_prefix("int") {
            if let Ok(parsed) = value.parse::<f64>() {
                intensity_power = parsed;
            }
        } else if let Some(value) = part.strip_prefix("weighted") {
            weighted = Some(value == "true");
        }
    }

    let family_label = match (family_kind, weighted) {
        (PrettyFamily::Cosine, _) => "Cosine",
        (PrettyFamily::ModifiedCosine, _) => "Modified cosine",
        (PrettyFamily::Entropy, Some(true)) => "Weighted entropy",
        (PrettyFamily::Entropy, Some(false)) => "Unweighted entropy",
        (PrettyFamily::ModifiedEntropy, Some(true)) => "Weighted modified entropy",
        (PrettyFamily::ModifiedEntropy, Some(false)) => "Unweighted modified entropy",
        (PrettyFamily::Entropy | PrettyFamily::ModifiedEntropy, None) => return slug.to_string(),
    };

    let weight_expression = format_weight_expression(mz_power, intensity_power);
    if weight_expression.is_empty() {
        family_label.to_string()
    } else {
        format!("{family_label}, w ∝ {weight_expression}")
    }
}

/// Similarity-family discriminator used while pretty-printing a config slug.
enum PrettyFamily {
    /// Linear cosine similarity.
    Cosine,
    /// Modified linear cosine similarity (allows m/z shifts).
    ModifiedCosine,
    /// Entropy similarity.
    Entropy,
    /// Modified entropy similarity.
    ModifiedEntropy,
}

/// Render the weighted-peak product for one config, dropping zero-power terms.
fn format_weight_expression(mz_power: f64, intensity_power: f64) -> String {
    let mz_part = format_weight_term("m/z", mz_power);
    let intensity_part = format_weight_term("intensity", intensity_power);
    match (mz_part.is_empty(), intensity_part.is_empty()) {
        (true, true) => String::new(),
        (true, false) => intensity_part,
        (false, true) => mz_part,
        (false, false) => format!("{mz_part} · {intensity_part}"),
    }
}

/// Render one factor of the weight product, omitting `^1` and `^0` exponents.
fn format_weight_term(name: &str, exponent: f64) -> String {
    if exponent == 0.0 {
        String::new()
    } else if (exponent - 1.0).abs() < f64::EPSILON {
        name.to_string()
    } else {
        format!("{name}^{}", format_exponent(exponent))
    }
}

/// Pretty-print a non-trivial exponent, substituting Unicode vulgar-fraction
/// glyphs when the value matches a common rational number.
///
/// `DejaVu Sans` renders these single codepoints as properly stacked fractions
/// (½, ¼, ⅔, …), giving LaTeX-quality typography in the figure title without
/// any external math renderer.
fn format_exponent(value: f64) -> String {
    const FRACTION_GLYPHS: &[(f64, &str)] = &[
        (1.0 / 8.0, "⅛"),
        (1.0 / 6.0, "⅙"),
        (1.0 / 5.0, "⅕"),
        (1.0 / 4.0, "¼"),
        (1.0 / 3.0, "⅓"),
        (3.0 / 8.0, "⅜"),
        (2.0 / 5.0, "⅖"),
        (1.0 / 2.0, "½"),
        (3.0 / 5.0, "⅗"),
        (5.0 / 8.0, "⅝"),
        (2.0 / 3.0, "⅔"),
        (3.0 / 4.0, "¾"),
        (4.0 / 5.0, "⅘"),
        (5.0 / 6.0, "⅚"),
        (7.0 / 8.0, "⅞"),
    ];
    const EXPONENT_FRACTION_TOLERANCE: f64 = 1.0e-6;
    for (target, glyph) in FRACTION_GLYPHS {
        if (value - target).abs() < EXPONENT_FRACTION_TOLERANCE {
            return (*glyph).to_string();
        }
    }
    format!("{value}")
}

/// Return a filesystem-safe path component.
pub fn sanitize_path_component(raw: &str) -> PathBuf {
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
pub fn plotters_error<Error: std::fmt::Debug>(error: Error) -> anyhow::Error {
    anyhow!("{error:?}")
}

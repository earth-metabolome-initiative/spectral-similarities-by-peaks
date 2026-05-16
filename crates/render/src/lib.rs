//! WASM-compatible rendering for spectral-similarity heatmaps.
//!
//! Provides the pure (no file I/O, no environment access) rendering pipeline
//! used by both the CLI (which writes SVG / PNG to disk) and the browser
//! viewer (which renders SVG into the DOM).
//!
//! The font face is embedded via `include_bytes!`, so both the CLI and the
//! WASM target can call [`ensure_font`] without filesystem access.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;

#[cfg(not(target_arch = "wasm32"))]
use anyhow::bail;
use anyhow::{Result, anyhow};
use colorous::Gradient;
use ndarray::{Array2, ArrayView2, ArrayView3, Axis};
#[cfg(not(target_arch = "wasm32"))]
use plotters::style::{FontStyle, register_font};
use plotters::{
    coord::Shift,
    prelude::{
        BLACK, ChartBuilder, DrawingArea, DrawingBackend, IntoDrawingArea, PathElement, RGBColor,
        Rectangle, SVGBackend, SeriesLabelPosition, WHITE,
    },
    style::{
        Color, IntoFont, ShapeStyle,
        text_anchor::{HPos, Pos, VPos},
    },
};

/// Linear scale factor applied to every pixel dimension.
///
/// Canvas size, margins, font sizes, and stroke widths all multiply by this.
/// Doubling this constant doubles the rendered PNG resolution while
/// preserving the visual layout; the SVG output is also drawn larger but is
/// unaffected visually since it stays vector-based.
pub const RENDER_SCALE: u32 = 2;

/// Signed counterpart of [`RENDER_SCALE`] for the few places that take `i32`
/// (label coordinates, legend bar length).
#[allow(clippy::cast_possible_wrap)]
pub const RENDER_SCALE_I32: i32 = RENDER_SCALE as i32;

/// Width of each rendered heatmap image in pixels.
pub const IMAGE_WIDTH: u32 = 1_000 * RENDER_SCALE;
/// Height of each rendered heatmap image in pixels.
pub const IMAGE_HEIGHT: u32 = 900 * RENDER_SCALE;
/// Width reserved for the chart area before the colorbar.
pub const CHART_AREA_WIDTH: u32 = 860 * RENDER_SCALE;
/// Number of colored rectangles used to draw the colorbar.
const COLORBAR_STEPS: usize = 256;
/// Fallback color for non-finite matrix entries.
const NON_FINITE_COLOR: RGBColor = RGBColor(180, 180, 180);
/// Lower bound used when logarithmic scales encounter subnormal values.
const LOG_MINIMUM_POSITIVE: f64 = f64::MIN_POSITIVE;
/// Linear neighborhood around zero for signed logarithmic diverging scales.
const SIGNED_LOG_LINEAR_FRACTION: f64 = 1.0e-3;

/// Horizontal padding between the right edge of the colorbar panel and the
/// right-anchored colorbar label.
const COLORBAR_LABEL_RIGHT_PAD: i32 = 16 * RENDER_SCALE_I32;
/// Vertical position (pixels from the top of the colorbar panel) for the
/// colorbar label's vertical center. Sits just above the colorbar top edge,
/// which is at `margin_top = 90` pixels of the same panel.
const COLORBAR_LABEL_TOP: i32 = 70 * RENDER_SCALE_I32;

/// Embedded font bytes used for native text rendering.
///
/// On wasm32 the browser renders SVG text via its own font subsystem, so the
/// embedded bytes are unused there.
#[cfg(not(target_arch = "wasm32"))]
const EMBEDDED_FONT: &[u8] = include_bytes!("../assets/DejaVuSans.ttf");

/// Cached result of registering the Plotters sans-serif font.
#[cfg(not(target_arch = "wasm32"))]
static FONT_REGISTRATION: OnceLock<std::result::Result<(), String>> = OnceLock::new();

/// Register the embedded sans-serif font with plotters.
///
/// On native targets this calls [`plotters::style::register_font`] once with
/// the embedded `DejaVu Sans` bytes. On wasm32 the function is a no-op: the
/// browser's font subsystem renders SVG `font-family="sans-serif"` text
/// directly without plotters needing the glyph data.
///
/// # Errors
///
/// Returns an error when the embedded font cannot be parsed (should be
/// impossible in practice since the bundled font is validated at build time).
#[cfg(not(target_arch = "wasm32"))]
pub fn ensure_font() -> Result<()> {
    match FONT_REGISTRATION.get_or_init(register_embedded_font) {
        Ok(()) => Ok(()),
        Err(message) => bail!("{message}"),
    }
}

/// WASM-target no-op variant of [`ensure_font`]. The browser handles fonts.
#[cfg(target_arch = "wasm32")]
#[allow(clippy::missing_errors_doc, clippy::missing_const_for_fn)]
pub fn ensure_font() -> Result<()> {
    Ok(())
}

/// Attempt to register the embedded font with plotters' `sans-serif` family.
#[cfg(not(target_arch = "wasm32"))]
fn register_embedded_font() -> std::result::Result<(), String> {
    register_font("sans-serif", FontStyle::Normal, EMBEDDED_FONT)
        .map_err(|_| "embedded DejaVu Sans is not a valid TrueType/OpenType font".to_string())
}

/// Convert a plotters backend error into an [`anyhow::Error`].
pub fn plotters_error<Error: std::fmt::Debug>(error: Error) -> anyhow::Error {
    anyhow!("{error:?}")
}

/// Minimum cell-distance from the diagonal for a curve to count as "converged".
///
/// Curves whose extreme endpoint is within this many cells of the main
/// diagonal are treated as still hugging the diagonal and not labeled with
/// an asymptote tick. The p-value α-contour at large samples never
/// separates from the diagonal, so its rightmost / topmost endpoint just
/// reflects where the data array ends; labeling that would be misleading.
pub const ASYMPTOTE_MIN_DIAGONAL_OFFSET: f64 = 15.0;

/// Bright pastel stroke colors cycled through, one per significance-threshold
/// contour. Chosen to read well against both viridis and red-blue palettes.
pub const CONTOUR_COLORS: &[RGBColor] = &[
    RGBColor(255, 99, 132),  // pastel red
    RGBColor(102, 187, 255), // pastel sky blue
    RGBColor(120, 220, 140), // pastel green
    RGBColor(255, 180, 90),  // pastel orange
    RGBColor(190, 130, 240), // pastel violet
];

/// Sample-size-independent "practical" thresholds for the KS-statistic contour overlay.
///
/// Drawn in this order so the lenient curve sits underneath and the strict
/// one on top. `0.10` is the data-drift literature's small/moderate
/// boundary, `0.05` is the negligible/small boundary, and `0.01` is a
/// tighter "practically detectable" boundary that hugs the diagonal where
/// the CDFs are nearly identical.
pub const KS_STATISTIC_THRESHOLDS: &[f64] = &[0.10, 0.05, 0.01];

/// Peak-count axis tick positions kept on every distribution heatmap.
pub const HEATMAP_AXIS_TICKS: &[i32] = &[0, 32, 64, 96, 128];

/// Metric matrix and rendering parameters.
pub struct HeatmapMetric<'a> {
    /// Stable file stem for the metric.
    pub name: &'static str,
    /// Human-readable title for the metric.
    pub title: &'static str,
    /// Short label shown above the colorbar.
    pub colorbar_label: &'static str,
    /// Matrix values for one similarity config.
    pub values: ArrayView2<'a, f64>,
    /// Value scale used to normalize colors.
    pub scale: HeatmapScale,
    /// Color palette used to render values.
    pub palette: Gradient,
    /// Canonical value to render on the main diagonal regardless of the
    /// stored matrix entry. A distribution compared against itself has
    /// `mean_delta=0`, `ks_statistic=0`, `ks_pvalue_asymptotic=1`, and
    /// `wasserstein_1d=0`; forcing this value here guarantees the diagonal
    /// always shows identity regardless of any upstream computation bug or
    /// pre-fix npz contents.
    pub diagonal_value: f64,
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

/// Value scale used to normalize one metric.
#[derive(Clone, Copy, PartialEq)]
pub enum HeatmapScale {
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
    #[must_use]
    pub fn sequential(minimum: f64, maximum: f64) -> Self {
        let maximum = if maximum.is_finite() && maximum > minimum {
            maximum
        } else {
            minimum + 1.0
        };
        Self::Sequential { minimum, maximum }
    }

    /// Create a logarithmic sequential scale with a nonzero positive floor.
    #[must_use]
    pub fn log_sequential(minimum: f64, positive_minimum: f64, maximum: f64) -> Self {
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
    #[must_use]
    pub fn signed_log_diverging_zero(maximum_abs: f64) -> Self {
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
    #[must_use]
    pub fn signed_diverging_zero(maximum_abs: f64) -> Self {
        let maximum_abs = if maximum_abs.is_finite() && maximum_abs > 0.0 {
            maximum_abs
        } else {
            1.0
        };
        Self::SignedDivergingZero { maximum_abs }
    }

    /// Normalize a value to a `[0, 1]` palette coordinate.
    #[must_use]
    pub fn normalize(self, value: f64) -> f64 {
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
    #[must_use]
    pub fn value_at(self, position: f64) -> f64 {
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

/// One line segment of a contour polyline, expressed in chart coordinates
/// (column-axis x, row-axis y) where the sample at `grid[(r, c)]` lives at
/// `(c + 0.5, r + 0.5)`.
pub type ContourSegment = ((f64, f64), (f64, f64));

/// One contour curve rendered on top of every heatmap.
pub struct ContourCurve {
    /// Pre-formatted legend label for this curve.
    pub label: String,
    /// Unordered marching-squares segments in chart coordinates.
    pub segments: Vec<ContourSegment>,
    /// Stroke color used for this curve. The axis-tick labels are drawn in
    /// the same color.
    pub color: RGBColor,
    /// X-axis tick positions: the curve's x-coordinate at its topmost
    /// endpoint, shifted by +1 for the chart's cell-offset convention.
    pub x_axis_ticks: Vec<i32>,
    /// Y-axis tick positions: the curve's y-coordinate at its rightmost
    /// endpoint.
    pub y_axis_ticks: Vec<i32>,
}

/// Per-config contour set shared across all 8 metric heatmaps.
#[derive(Clone, Copy)]
pub struct ContourOverlay<'a> {
    /// One curve per requested significance level / effect-size threshold.
    pub curves: &'a [ContourCurve],
}

/// Draw one heatmap into a concrete backend.
///
/// `dataset_label` is prefixed to the chart caption when supplied (e.g.,
/// `"harmonized-full"`). Pass `None` to omit the dataset segment entirely.
///
/// # Errors
///
/// Returns an error if the underlying plotters backend fails (e.g., I/O
/// error in an SVG/PNG writer, or unsupported drawing primitive).
pub fn draw_heatmap<Backend>(
    root: &DrawingArea<Backend, Shift>,
    config: &str,
    metric: &HeatmapMetric<'_>,
    overlay: &ContourOverlay<'_>,
    dataset_label: Option<&str>,
) -> Result<()>
where
    Backend: DrawingBackend,
    Backend::ErrorType: std::fmt::Debug,
{
    root.fill(&WHITE).map_err(plotters_error)?;
    let (chart_area, colorbar_area) = root.split_horizontally(CHART_AREA_WIDTH);
    draw_matrix(&chart_area, config, metric, overlay, dataset_label)?;
    draw_colorbar(&colorbar_area, metric)?;
    root.present().map_err(plotters_error)
}

/// Compose the chart caption from an optional dataset, config slug, and metric.
fn compose_caption(dataset_label: Option<&str>, config: &str, metric_title: &str) -> String {
    let pretty = pretty_config_title(config);
    dataset_label.map_or_else(
        || format!("{pretty} · {metric_title}"),
        |label| format!("{label} · {pretty} · {metric_title}"),
    )
}

/// Draw the main matrix panel.
#[allow(clippy::too_many_lines, clippy::similar_names)]
fn draw_matrix<Backend>(
    area: &DrawingArea<Backend, Shift>,
    config: &str,
    metric: &HeatmapMetric<'_>,
    overlay: &ContourOverlay<'_>,
    dataset_label: Option<&str>,
) -> Result<()>
where
    Backend: DrawingBackend,
    Backend::ErrorType: std::fmt::Debug,
{
    let rows = metric.values.nrows();
    let columns = metric.values.ncols();
    let x_end = (columns + 1) as f64;
    let y_end = (rows + 1) as f64;
    let mut chart = ChartBuilder::on(area)
        .caption(
            compose_caption(dataset_label, config, metric.title),
            ("sans-serif", 24 * RENDER_SCALE),
        )
        .margin(22 * RENDER_SCALE)
        .x_label_area_size(48 * RENDER_SCALE)
        .y_label_area_size(58 * RENDER_SCALE)
        .build_cartesian_2d(0.0_f64..x_end, 0.0_f64..y_end)
        .map_err(plotters_error)?;

    // The plotters default mesh renders only the canonical axis ticks in
    // black. Per-curve crossing ticks are drawn manually below in the
    // matching curve color. To avoid a stale black tick peeking through
    // when a curve's asymptote happens to land exactly on a default tick
    // (e.g., D-curve at 32), we collect the set of default tick values
    // that will be overlaid by a colored draw and suppress them in the
    // appropriate axis formatter.
    let mut x_axis_overlaid: Vec<i32> = Vec::new();
    let mut y_axis_overlaid: Vec<i32> = Vec::new();
    for curve in overlay.curves {
        for &tick in &curve.x_axis_ticks {
            if HEATMAP_AXIS_TICKS.contains(&tick) && !x_axis_overlaid.contains(&tick) {
                x_axis_overlaid.push(tick);
            }
        }
        for &tick in &curve.y_axis_ticks {
            if HEATMAP_AXIS_TICKS.contains(&tick) && !y_axis_overlaid.contains(&tick) {
                y_axis_overlaid.push(tick);
            }
        }
    }
    let x_overlaid = &x_axis_overlaid;
    let y_overlaid = &y_axis_overlaid;
    let x_tick_label = |value: &f64| {
        let rounded = value.round();
        if (value - rounded).abs() < 1.0e-6 {
            let as_int = rounded as i32;
            if HEATMAP_AXIS_TICKS.contains(&as_int) && !x_overlaid.contains(&as_int) {
                return as_int.to_string();
            }
        }
        String::new()
    };
    let y_tick_label = |value: &f64| {
        let rounded = value.round();
        if (value - rounded).abs() < 1.0e-6 {
            let as_int = rounded as i32;
            if HEATMAP_AXIS_TICKS.contains(&as_int) && !y_overlaid.contains(&as_int) {
                return as_int.to_string();
            }
        }
        String::new()
    };
    let x_end_usize = usize::try_from(x_end as i64).unwrap_or(usize::MAX);
    let y_end_usize = usize::try_from(y_end as i64).unwrap_or(usize::MAX);
    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("Top peaks retained")
        .y_desc("Top peaks retained")
        .x_labels(x_end_usize)
        .y_labels(y_end_usize)
        .x_label_formatter(&x_tick_label)
        .y_label_formatter(&y_tick_label)
        .axis_desc_style(("sans-serif", 20 * RENDER_SCALE))
        .label_style(("sans-serif", 16 * RENDER_SCALE))
        .draw()
        .map_err(plotters_error)?;

    chart
        .draw_series(matrix_cells(metric))
        .map_err(plotters_error)?;

    // Marching-squares output places samples at cell centers (c+0.5, r+0.5)
    // but `matrix_cells` draws sample (r, c) at chart rect [c+1, c+2] × [r+1, r+2].
    // Shift each contour endpoint by +1 in both axes to land on the visual
    // cell centers (c + 1.5, r + 1.5) before drawing.
    //
    // Iteration order: lenient (largest α, outermost contour) first so the
    // stricter curves overlay on top of it.
    let mut has_legend_entry = false;
    for curve in overlay.curves.iter().rev() {
        if curve.segments.is_empty() {
            continue;
        }
        let style = ShapeStyle {
            color: curve.color.to_rgba(),
            filled: false,
            stroke_width: 2 * RENDER_SCALE,
        };
        let mut series_iter = curve
            .segments
            .iter()
            .map(|&((x1, y1), (x2, y2))| vec![(x1 + 1.0, y1 + 1.0), (x2 + 1.0, y2 + 1.0)]);
        let Some(first) = series_iter.next() else {
            continue;
        };
        let label = curve.label.clone();
        chart
            .draw_series(std::iter::once(PathElement::new(first, style)))
            .map_err(plotters_error)?
            .label(label)
            .legend(move |(x, y)| {
                // Plotters hands us the vertical centre of the font bounding
                // box; that sits a little above the visible x-height middle,
                // so nudge the stroke down a couple of pixels so it lines up
                // with the body of the label text.
                let line_y = y + 2 * RENDER_SCALE_I32;
                PathElement::new(
                    vec![(x, line_y), (x + 24 * RENDER_SCALE_I32, line_y)],
                    style,
                )
            });
        for points in series_iter {
            chart
                .draw_series(std::iter::once(PathElement::new(points, style)))
                .map_err(plotters_error)?;
        }
        has_legend_entry = true;
    }
    if has_legend_entry {
        // `Coordinate(x, y)` is interpreted by plotters as a pixel offset
        // from the plotting area's top-left, not the chart's outer origin,
        // so small positive values nudge the legend just inside the
        // top-left corner of the plot.
        let legend_x = 10 * RENDER_SCALE_I32;
        let legend_y = 5 * RENDER_SCALE_I32;
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::Coordinate(legend_x, legend_y))
            .background_style(WHITE.mix(0.82))
            .border_style(BLACK)
            .legend_area_size(30 * RENDER_SCALE)
            .label_font(("sans-serif", 14 * RENDER_SCALE).into_font())
            .draw()
            .map_err(plotters_error)?;
    }

    // Per-curve axis tick labels. Each curve contributes:
    //   - one x-axis label at its topmost endpoint's x position (where the
    //     curve reaches the top edge of the plot)
    //   - one y-axis label at its rightmost endpoint's y position (where
    //     the curve reaches the right edge of the plot)
    // Both labels are drawn in the curve's color so the connection is
    // immediate. Default ticks (0, 32, 64, 96, 128) remain black.
    let plot_pixels = chart.plotting_area().get_pixel_range();
    let (plot_x_min, plot_y_max) = (plot_pixels.0.start, plot_pixels.1.end);
    let x_label_y = plot_y_max + 6 * RENDER_SCALE_I32;
    let y_label_x = plot_x_min - 6 * RENDER_SCALE_I32;
    for curve in overlay.curves {
        let style_x = ("sans-serif", 16 * RENDER_SCALE)
            .into_font()
            .color(&curve.color)
            .pos(Pos::new(HPos::Center, VPos::Top));
        let style_y = ("sans-serif", 16 * RENDER_SCALE)
            .into_font()
            .color(&curve.color)
            .pos(Pos::new(HPos::Right, VPos::Center));
        for &tick in &curve.x_axis_ticks {
            // Always draw — when the asymptote tick coincides with a
            // default tick value (e.g., 32, 64), the colored label overlays
            // the plotters-default black one, effectively colorizing it.
            let (tick_px, _) = chart.backend_coord(&(f64::from(tick), 0.0));
            area.draw_text(&tick.to_string(), &style_x, (tick_px, x_label_y))
                .map_err(plotters_error)?;
        }
        for &tick in &curve.y_axis_ticks {
            let (_, tick_py) = chart.backend_coord(&(0.0, f64::from(tick)));
            area.draw_text(&tick.to_string(), &style_y, (y_label_x, tick_py))
                .map_err(plotters_error)?;
        }
    }
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
    let (area_width, _) = area.dim_in_pixel();
    let label_x = i32::try_from(area_width)?.saturating_sub(COLORBAR_LABEL_RIGHT_PAD);
    let label_style = ("sans-serif", 16 * RENDER_SCALE)
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
        .margin_left(4 * RENDER_SCALE)
        .margin_right(12 * RENDER_SCALE)
        .margin_top(90 * RENDER_SCALE)
        .margin_bottom(86 * RENDER_SCALE)
        .y_label_area_size(80 * RENDER_SCALE)
        .build_cartesian_2d(0.0_f64..1.0_f64, 0.0_f64..1.0_f64)
        .map_err(plotters_error)?;
    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_labels(0)
        .y_labels(7)
        .label_style(("sans-serif", 15 * RENDER_SCALE))
        .y_label_formatter(&|position| format_tick(metric.scale.value_at(*position)))
        .draw()
        .map_err(plotters_error)?;
    chart
        .draw_series(colorbar_cells(metric))
        .map_err(plotters_error)?;
    Ok(())
}

/// Return all colored matrix cells for a heatmap.
fn matrix_cells(metric: &HeatmapMetric<'_>) -> Vec<Rectangle<(f64, f64)>> {
    let mut cells = Vec::with_capacity(metric.values.len());
    for row in 0..metric.values.nrows() {
        for column in 0..metric.values.ncols() {
            let x0 = (column + 1) as f64;
            let y0 = (row + 1) as f64;
            let value = if row == column {
                metric.diagonal_value
            } else {
                metric.values[[row, column]]
            };
            cells.push(Rectangle::new(
                [(x0, y0), (x0 + 1.0, y0 + 1.0)],
                metric.color(value).filled(),
            ));
        }
    }
    cells
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

/// Format a colorbar tick value.
///
/// Only an exact `0.0` collapses to the literal `"0"`. Genuinely tiny values
/// (e.g. log-scale p-value ticks like `1e-100`) are not rounded to zero —
/// they flow into scientific notation so the colorbar stays readable across
/// many orders of magnitude.
#[must_use]
pub fn format_tick(value: f64) -> String {
    if value == 0.0 {
        "0".to_string()
    } else if (0.001..1_000.0).contains(&value.abs()) {
        format!("{value:.3}")
    } else {
        format!("{value:.2e}")
    }
}

/// Format a significance-level label for the heatmap legend.
///
/// Uses scientific notation for very small alphas, three decimals for the
/// mid range, and integer rendering for `1.0`.
#[must_use]
pub fn format_contour_label(alpha: f64) -> String {
    format!("α = {}", format_contour_value(alpha))
}

/// Format a `D` (KS-statistic) threshold label for the heatmap legend.
#[must_use]
pub fn format_d_label(threshold: f64) -> String {
    format!("D = {}", format_contour_value(threshold))
}

/// Shared numeric formatter for both `α` and `D` legend labels.
fn format_contour_value(value: f64) -> String {
    if value <= 0.0 {
        "0".to_string()
    } else if value < 0.01 {
        format!("{value:.2e}")
    } else if (value - 1.0).abs() < 1.0e-9 {
        "1".to_string()
    } else {
        format!("{value:.3}")
    }
}

/// Pretty-print a `SimilarityConfig::name()` slug as a human-readable title.
///
/// Drops terms whose exponent is 0 (they contribute the identity 1 to the
/// weight product) and omits the `^1` suffix when an exponent equals 1.
#[must_use]
pub fn pretty_config_title(slug: &str) -> String {
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

/// Pretty-print a non-trivial exponent.
///
/// Substitutes Unicode vulgar-fraction glyphs when the value matches a
/// common rational number.
#[must_use]
pub fn format_exponent(value: f64) -> String {
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

/// Compute the per-axis tick positions for one curve. Returns
/// `(x_axis_ticks, y_axis_ticks)` where:
/// - `x_axis_ticks` is the curve's x-coordinate at its topmost endpoint
///   (where it hits the top edge of the plot), shown only on the x-axis;
/// - `y_axis_ticks` is the curve's y-coordinate at its rightmost endpoint,
///   shown only on the y-axis.
///
/// Returns empty vectors for curves that never separate sufficiently from
/// the diagonal.
#[must_use]
pub fn curve_edge_ticks(segments: &[ContourSegment]) -> (Vec<i32>, Vec<i32>) {
    if segments.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut max_x = f64::NEG_INFINITY;
    let mut y_at_max_x = 0.0_f64;
    let mut max_y = f64::NEG_INFINITY;
    let mut x_at_max_y = 0.0_f64;
    for &((x1, y1), (x2, y2)) in segments {
        for (x, y) in [(x1, y1), (x2, y2)] {
            if x > max_x {
                max_x = x;
                y_at_max_x = y;
            }
            if y > max_y {
                max_y = y;
                x_at_max_y = x;
            }
        }
    }
    let mut x_axis_ticks = Vec::new();
    let mut y_axis_ticks = Vec::new();
    if max_x - y_at_max_x > ASYMPTOTE_MIN_DIAGONAL_OFFSET {
        y_axis_ticks.push((y_at_max_x.round() as i32) + 1);
    }
    if max_y - x_at_max_y > ASYMPTOTE_MIN_DIAGONAL_OFFSET {
        x_axis_ticks.push((x_at_max_y.round() as i32) + 1);
    }
    (x_axis_ticks, y_axis_ticks)
}

/// Return the peak-count value at which the `threshold`-level contour of a
/// single config's KS-statistic grid hits the right edge of the plot.
#[must_use]
pub fn ks_statistic_asymptote(grid: &ArrayView2<f64>, threshold: f64) -> Option<i32> {
    let mut owned: Array2<f64> = grid.to_owned();
    mask_main_diagonal(&mut owned);
    let segments = extract_contour(&owned.view(), threshold);
    if segments.is_empty() {
        return None;
    }
    let mut max_x = f64::NEG_INFINITY;
    let mut y_at_max_x = 0.0_f64;
    for &((x1, y1), (x2, y2)) in &segments {
        for (x, y) in [(x1, y1), (x2, y2)] {
            if x > max_x {
                max_x = x;
                y_at_max_x = y;
            }
        }
    }
    if max_x - y_at_max_x > ASYMPTOTE_MIN_DIAGONAL_OFFSET {
        Some((y_at_max_x.round() as i32) + 1)
    } else {
        None
    }
}

/// Set every self-comparison cell (`row == col`) of `grid` to NaN, so
/// [`extract_contour`] skips every 2×2 quad that touches the main diagonal.
pub fn mask_main_diagonal(grid: &mut Array2<f64>) {
    let len = grid.nrows().min(grid.ncols());
    for k in 0..len {
        grid[(k, k)] = f64::NAN;
    }
}

/// Trace the `alpha`-level contour of a 2D grid using marching squares.
///
/// Uses linear edge interpolation. Returns an unordered list of line
/// segments; segments in ambiguous saddle cells are resolved against the
/// cell centroid.
#[must_use]
pub fn extract_contour(grid: &ArrayView2<f64>, alpha: f64) -> Vec<ContourSegment> {
    let mut segments = Vec::new();
    let (n_rows, n_cols) = grid.dim();
    if n_rows < 2 || n_cols < 2 {
        return segments;
    }
    for r in 0..n_rows - 1 {
        for c in 0..n_cols - 1 {
            let tl = grid[(r, c)];
            let tr = grid[(r, c + 1)];
            let br = grid[(r + 1, c + 1)];
            let bl = grid[(r + 1, c)];
            if !(tl.is_finite() && tr.is_finite() && br.is_finite() && bl.is_finite()) {
                continue;
            }
            let xl = c as f64 + 0.5;
            let xr = (c + 1) as f64 + 0.5;
            let yt = r as f64 + 0.5;
            let yb = (r + 1) as f64 + 0.5;

            let mut case: u8 = 0;
            if tl >= alpha {
                case |= 1;
            }
            if tr >= alpha {
                case |= 2;
            }
            if br >= alpha {
                case |= 4;
            }
            if bl >= alpha {
                case |= 8;
            }

            let interp = |a: f64, b: f64| {
                let denom = b - a;
                if denom == 0.0 {
                    0.5
                } else {
                    ((alpha - a) / denom).clamp(0.0, 1.0)
                }
            };
            let top = || (xl + interp(tl, tr) * (xr - xl), yt);
            let right = || (xr, yt + interp(tr, br) * (yb - yt));
            let bottom = || (xl + interp(bl, br) * (xr - xl), yb);
            let left = || (xl, yt + interp(tl, bl) * (yb - yt));

            match case {
                0 | 15 => {}
                1 | 14 => segments.push((left(), top())),
                2 | 13 => segments.push((top(), right())),
                3 | 12 => segments.push((left(), right())),
                4 | 11 => segments.push((right(), bottom())),
                6 | 9 => segments.push((top(), bottom())),
                7 | 8 => segments.push((left(), bottom())),
                5 => {
                    let center = (tl + tr + br + bl) * 0.25;
                    if center >= alpha {
                        segments.push((top(), right()));
                        segments.push((left(), bottom()));
                    } else {
                        segments.push((left(), top()));
                        segments.push((right(), bottom()));
                    }
                }
                10 => {
                    let center = (tl + tr + br + bl) * 0.25;
                    if center >= alpha {
                        segments.push((left(), top()));
                        segments.push((right(), bottom()));
                    } else {
                        segments.push((top(), right()));
                        segments.push((left(), bottom()));
                    }
                }
                _ => unreachable!(),
            }
        }
    }
    segments
}

/// Derive the extra axis tick positions where the contour reaches each plot edge.
///
/// Rounds to the nearest integer. Returns at most four ticks (min-x, max-x,
/// min-y, max-y), sorted and deduplicated.
#[must_use]
pub fn axis_crossings(segments: &[ContourSegment]) -> Vec<i32> {
    if segments.is_empty() {
        return Vec::new();
    }
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for &((x1, y1), (x2, y2)) in segments {
        for x in [x1, x2] {
            if x < min_x {
                min_x = x;
            }
            if x > max_x {
                max_x = x;
            }
        }
        for y in [y1, y2] {
            if y < min_y {
                min_y = y;
            }
            if y > max_y {
                max_y = y;
            }
        }
    }
    let mut ticks: Vec<i32> = [min_x, max_x, min_y, max_y]
        .iter()
        .map(|value| value.round() as i32)
        .collect();
    ticks.sort_unstable();
    ticks.dedup();
    ticks
}

// ---------------------------------------------------------------------------
// High-level dataset rendering API
// ---------------------------------------------------------------------------

/// Borrowed views into one dataset's four `(config × peak × peak)` grids.
///
/// Cheap to copy because every field is itself a borrowed `ArrayView3`.
#[derive(Clone, Copy)]
pub struct GridViews<'a> {
    /// Δμ (mean delta) grid.
    pub mean_delta: ArrayView3<'a, f64>,
    /// KS statistic grid.
    pub ks_statistic: ArrayView3<'a, f64>,
    /// Asymptotic KS p-value grid.
    pub ks_pvalue_asymptotic: ArrayView3<'a, f64>,
    /// 1D Wasserstein distance grid.
    pub wasserstein_1d: ArrayView3<'a, f64>,
}

/// One of the eight rendered heatmap variants per similarity config.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Metric {
    /// Δμ on a linear diverging palette.
    MeanDeltaLinear,
    /// Δμ on a signed-log diverging palette.
    MeanDeltaLog,
    /// KS statistic on a linear sequential palette.
    KsStatisticLinear,
    /// KS statistic on a log sequential palette.
    KsStatisticLog,
    /// Asymptotic KS p-value on a linear sequential palette.
    KsPvalueAsymptoticLinear,
    /// Asymptotic KS p-value on a log sequential palette.
    KsPvalueAsymptoticLog,
    /// 1D Wasserstein distance on a linear sequential palette.
    Wasserstein1dLinear,
    /// 1D Wasserstein distance on a log sequential palette.
    Wasserstein1dLog,
}

impl Metric {
    /// All eight metric variants in canonical render order.
    pub const ALL: [Self; 8] = [
        Self::MeanDeltaLinear,
        Self::MeanDeltaLog,
        Self::KsStatisticLinear,
        Self::KsStatisticLog,
        Self::KsPvalueAsymptoticLinear,
        Self::KsPvalueAsymptoticLog,
        Self::Wasserstein1dLinear,
        Self::Wasserstein1dLog,
    ];

    /// Stable file-stem / URL-slug for this metric.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::MeanDeltaLinear => "mean_delta_linear",
            Self::MeanDeltaLog => "mean_delta_log",
            Self::KsStatisticLinear => "ks_statistic_linear",
            Self::KsStatisticLog => "ks_statistic_log",
            Self::KsPvalueAsymptoticLinear => "ks_pvalue_asymptotic_linear",
            Self::KsPvalueAsymptoticLog => "ks_pvalue_asymptotic_log",
            Self::Wasserstein1dLinear => "wasserstein_1d_linear",
            Self::Wasserstein1dLog => "wasserstein_1d_log",
        }
    }

    /// Human-readable heatmap caption.
    #[must_use]
    pub const fn title(self) -> &'static str {
        match self {
            Self::MeanDeltaLinear => "Δ mean (linear)",
            Self::MeanDeltaLog => "Δ mean (signed log)",
            Self::KsStatisticLinear => "KS statistic (linear)",
            Self::KsStatisticLog => "KS statistic (log)",
            Self::KsPvalueAsymptoticLinear => "KS p-value (linear)",
            Self::KsPvalueAsymptoticLog => "KS p-value (log)",
            Self::Wasserstein1dLinear => "Wasserstein (linear)",
            Self::Wasserstein1dLog => "Wasserstein (log)",
        }
    }

    /// Short label shown above the colorbar.
    #[must_use]
    pub const fn colorbar_label(self) -> &'static str {
        match self {
            Self::MeanDeltaLinear | Self::MeanDeltaLog => "mean delta",
            Self::KsStatisticLinear | Self::KsStatisticLog => "KS",
            Self::KsPvalueAsymptoticLinear | Self::KsPvalueAsymptoticLog => "p-value",
            Self::Wasserstein1dLinear | Self::Wasserstein1dLog => "Wasserstein",
        }
    }

    /// Color palette for this metric.
    #[must_use]
    pub const fn palette(self) -> Gradient {
        match self {
            Self::MeanDeltaLinear | Self::MeanDeltaLog => colorous::RED_BLUE,
            _ => colorous::VIRIDIS,
        }
    }

    /// Canonical value to draw on the main diagonal (self-comparison).
    #[must_use]
    pub const fn diagonal_value(self) -> f64 {
        match self {
            Self::KsPvalueAsymptoticLinear | Self::KsPvalueAsymptoticLog => 1.0,
            _ => 0.0,
        }
    }
}

/// Precomputed value scales for one dataset's four grids.
#[derive(Clone, Copy, PartialEq)]
pub struct Scales {
    /// Linear diverging scale for mean-delta heatmaps, centered at zero.
    pub mean_delta_linear: HeatmapScale,
    /// Signed logarithmic diverging scale for mean-delta heatmaps.
    pub mean_delta_log: HeatmapScale,
    /// Linear sequential scale for KS statistic heatmaps.
    pub ks_statistic_linear: HeatmapScale,
    /// Logarithmic sequential scale for KS statistic heatmaps.
    pub ks_statistic_log: HeatmapScale,
    /// Linear sequential scale for asymptotic KS p-value heatmaps.
    pub ks_pvalue_asymptotic_linear: HeatmapScale,
    /// Logarithmic sequential scale for asymptotic KS p-value heatmaps.
    pub ks_pvalue_asymptotic_log: HeatmapScale,
    /// Linear sequential scale for Wasserstein heatmaps.
    pub wasserstein_1d_linear: HeatmapScale,
    /// Logarithmic sequential scale for Wasserstein heatmaps.
    pub wasserstein_1d_log: HeatmapScale,
}

impl Scales {
    /// Compute global per-metric scales by scanning every cell of each grid.
    #[must_use]
    pub fn from_grids(grids: &GridViews<'_>) -> Self {
        let mean_delta_abs = max_abs_view3(&grids.mean_delta);
        let ks_min = finite_min_view3(&grids.ks_statistic).unwrap_or(0.0);
        let ks_pos_min = finite_positive_min_view3(&grids.ks_statistic).unwrap_or(1.0);
        let ks_max = finite_max_view3(&grids.ks_statistic).unwrap_or(1.0);
        let pv_min = finite_min_view3(&grids.ks_pvalue_asymptotic).unwrap_or(0.0);
        let pv_pos_min = finite_positive_min_view3(&grids.ks_pvalue_asymptotic).unwrap_or(1.0);
        let pv_max = finite_max_view3(&grids.ks_pvalue_asymptotic).unwrap_or(1.0);
        let ws_min = finite_min_view3(&grids.wasserstein_1d).unwrap_or(0.0);
        let ws_pos_min = finite_positive_min_view3(&grids.wasserstein_1d).unwrap_or(1.0);
        let ws_max = finite_max_view3(&grids.wasserstein_1d).unwrap_or(1.0);
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

/// Return the maximum absolute finite value across a 3D grid.
fn max_abs_view3(values: &ArrayView3<'_, f64>) -> f64 {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(f64::abs)
        .fold(0.0, f64::max)
}

/// Return the minimum finite value across a 3D grid, if any.
fn finite_min_view3(values: &ArrayView3<'_, f64>) -> Option<f64> {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .reduce(f64::min)
}

/// Return the maximum finite value across a 3D grid, if any.
fn finite_max_view3(values: &ArrayView3<'_, f64>) -> Option<f64> {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .reduce(f64::max)
}

/// Return the smallest finite positive value across a 3D grid, if any.
fn finite_positive_min_view3(values: &ArrayView3<'_, f64>) -> Option<f64> {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .reduce(f64::min)
}

/// Build the [`HeatmapMetric`] for one `(metric, config_index)` slice using
/// precomputed [`Scales`].
#[must_use]
pub fn metric_view<'a>(
    metric: Metric,
    grids: GridViews<'a>,
    scales: &Scales,
    config_index: usize,
) -> HeatmapMetric<'a> {
    let (values, scale) = match metric {
        Metric::MeanDeltaLinear => (
            grids.mean_delta.index_axis_move(Axis(0), config_index),
            scales.mean_delta_linear,
        ),
        Metric::MeanDeltaLog => (
            grids.mean_delta.index_axis_move(Axis(0), config_index),
            scales.mean_delta_log,
        ),
        Metric::KsStatisticLinear => (
            grids.ks_statistic.index_axis_move(Axis(0), config_index),
            scales.ks_statistic_linear,
        ),
        Metric::KsStatisticLog => (
            grids.ks_statistic.index_axis_move(Axis(0), config_index),
            scales.ks_statistic_log,
        ),
        Metric::KsPvalueAsymptoticLinear => (
            grids
                .ks_pvalue_asymptotic
                .index_axis_move(Axis(0), config_index),
            scales.ks_pvalue_asymptotic_linear,
        ),
        Metric::KsPvalueAsymptoticLog => (
            grids
                .ks_pvalue_asymptotic
                .index_axis_move(Axis(0), config_index),
            scales.ks_pvalue_asymptotic_log,
        ),
        Metric::Wasserstein1dLinear => (
            grids.wasserstein_1d.index_axis_move(Axis(0), config_index),
            scales.wasserstein_1d_linear,
        ),
        Metric::Wasserstein1dLog => (
            grids.wasserstein_1d.index_axis_move(Axis(0), config_index),
            scales.wasserstein_1d_log,
        ),
    };
    HeatmapMetric {
        name: metric.name(),
        title: metric.title(),
        colorbar_label: metric.colorbar_label(),
        values,
        scale,
        palette: metric.palette(),
        diagonal_value: metric.diagonal_value(),
    }
}

/// Build the per-config contour overlay shared across all 8 metric heatmaps.
///
/// Returns one curve per `threshold_alphas` entry (p-value contours), then
/// one curve per `d_thresholds` entry (KS-statistic effect-size contours).
/// Empty `d_thresholds` is allowed; the literature defaults are exposed as
/// [`KS_STATISTIC_THRESHOLDS`] for callers that want them.
#[must_use]
pub fn build_contour_curves(
    grids: &GridViews<'_>,
    config_index: usize,
    threshold_alphas: &[f64],
    d_thresholds: &[f64],
) -> Vec<ContourCurve> {
    let mut pvalue_grid: Array2<f64> = grids
        .ks_pvalue_asymptotic
        .index_axis(Axis(0), config_index)
        .to_owned();
    mask_main_diagonal(&mut pvalue_grid);
    let mut curves: Vec<ContourCurve> = threshold_alphas
        .iter()
        .enumerate()
        .map(|(index, &alpha)| {
            let segments = extract_contour(&pvalue_grid.view(), alpha);
            let (x_axis_ticks, y_axis_ticks) = curve_edge_ticks(&segments);
            ContourCurve {
                label: format_contour_label(alpha),
                segments,
                color: CONTOUR_COLORS[index % CONTOUR_COLORS.len()],
                x_axis_ticks,
                y_axis_ticks,
            }
        })
        .collect();

    let mut ks_grid: Array2<f64> = grids
        .ks_statistic
        .index_axis(Axis(0), config_index)
        .to_owned();
    mask_main_diagonal(&mut ks_grid);
    for &threshold in d_thresholds {
        let segments = extract_contour(&ks_grid.view(), threshold);
        if !segments.is_empty() {
            let (x_axis_ticks, y_axis_ticks) = curve_edge_ticks(&segments);
            curves.push(ContourCurve {
                label: format_d_label(threshold),
                segments,
                color: CONTOUR_COLORS[curves.len() % CONTOUR_COLORS.len()],
                x_axis_ticks,
                y_axis_ticks,
            });
        }
    }
    curves
}

/// Render one heatmap into an SVG `String`. The font is registered on first
/// call via [`ensure_font`]; on wasm32 this is a no-op and the browser
/// renders text from `font-family="sans-serif"`.
///
/// # Errors
///
/// Returns an error when the font registration fails (native only) or any
/// plotters call fails.
#[allow(clippy::too_many_arguments)]
pub fn render_cell_svg(
    config: &str,
    grids: GridViews<'_>,
    scales: &Scales,
    config_index: usize,
    metric: Metric,
    threshold_alphas: &[f64],
    d_thresholds: &[f64],
    dataset_label: Option<&str>,
) -> Result<String> {
    ensure_font()?;
    let curves = build_contour_curves(&grids, config_index, threshold_alphas, d_thresholds);
    let overlay = ContourOverlay { curves: &curves };
    let metric_view = metric_view(metric, grids, scales, config_index);
    let mut svg = String::new();
    {
        let root =
            SVGBackend::with_string(&mut svg, (IMAGE_WIDTH, IMAGE_HEIGHT)).into_drawing_area();
        draw_heatmap(&root, config, &metric_view, &overlay, dataset_label)?;
    }
    Ok(svg)
}

#[cfg(test)]
mod tests {
    use super::{ContourSegment, axis_crossings, extract_contour};
    use ndarray::{Array2, array};

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1.0e-9
    }

    #[test]
    fn extract_contour_returns_empty_when_all_corners_above() {
        let grid = array![[1.0, 1.0], [1.0, 1.0]];
        assert!(extract_contour(&grid.view(), 0.5).is_empty());
    }

    #[test]
    fn extract_contour_returns_empty_when_all_corners_below() {
        let grid = array![[0.0, 0.0], [0.0, 0.0]];
        assert!(extract_contour(&grid.view(), 0.5).is_empty());
    }

    #[test]
    fn extract_contour_horizontal_split_emits_one_horizontal_segment() {
        let grid = array![[1.0, 1.0], [0.0, 0.0]];
        let segments = extract_contour(&grid.view(), 0.5);
        assert_eq!(segments.len(), 1);
        let ((x1, y1), (x2, y2)) = segments[0];
        let (xa, xb) = if x1 < x2 { (x1, x2) } else { (x2, x1) };
        assert!(approx_eq(xa, 0.5));
        assert!(approx_eq(xb, 1.5));
        assert!(approx_eq(y1, 1.0));
        assert!(approx_eq(y2, 1.0));
    }

    #[test]
    fn extract_contour_isolated_corner_emits_one_corner_segment() {
        let grid = array![[1.0, 0.0], [0.0, 0.0]];
        let segments = extract_contour(&grid.view(), 0.5);
        assert_eq!(segments.len(), 1);
        let ((x1, y1), (x2, y2)) = segments[0];
        let endpoints = [(x1, y1), (x2, y2)];
        let on_left = endpoints
            .iter()
            .any(|(x, y)| approx_eq(*x, 0.5) && approx_eq(*y, 1.0));
        let on_top = endpoints
            .iter()
            .any(|(x, y)| approx_eq(*x, 1.0) && approx_eq(*y, 0.5));
        assert!(
            on_left && on_top,
            "expected (0.5, 1.0) and (1.0, 0.5), got {endpoints:?}"
        );
    }

    #[test]
    fn extract_contour_gaussian_bump_lies_on_expected_radius() {
        let n = 21usize;
        let mut grid = Array2::zeros((n, n));
        let sigma_sq = 16.0;
        for i in 0..n {
            for j in 0..n {
                let dy = (i as f64) - 10.0;
                let dx = (j as f64) - 10.0;
                grid[(i, j)] = ((-(dx * dx + dy * dy)) / sigma_sq).exp();
            }
        }
        let segments = extract_contour(&grid.view(), 0.5);
        assert!(!segments.is_empty(), "expected at least one segment");
        let expected_radius = (sigma_sq * std::f64::consts::LN_2).sqrt();
        let center = 10.5_f64;
        for ((x1, y1), (x2, y2)) in segments.iter().copied() {
            let r1 = (x1 - center).hypot(y1 - center);
            let r2 = (x2 - center).hypot(y2 - center);
            assert!(
                (r1 - expected_radius).abs() < 0.5,
                "endpoint ({x1}, {y1}) radius {r1} differs from {expected_radius}"
            );
            assert!(
                (r2 - expected_radius).abs() < 0.5,
                "endpoint ({x2}, {y2}) radius {r2} differs from {expected_radius}"
            );
        }
    }

    #[test]
    fn extract_contour_is_symmetric_for_symmetric_input() {
        let n = 5usize;
        let mut grid = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let d = ((i as f64) - (j as f64)).abs();
                grid[(i, j)] = (-d / 2.0).exp();
            }
        }
        let segments = extract_contour(&grid.view(), 0.5);
        let mut transposed = std::collections::HashSet::new();
        for ((x1, y1), (x2, y2)) in segments.iter().copied() {
            transposed.insert((
                ((y1 * 1.0e6).round() as i64, (x1 * 1.0e6).round() as i64),
                ((y2 * 1.0e6).round() as i64, (x2 * 1.0e6).round() as i64),
            ));
        }
        for ((x1, y1), (x2, y2)) in segments {
            let key = (
                ((x1 * 1.0e6).round() as i64, (y1 * 1.0e6).round() as i64),
                ((x2 * 1.0e6).round() as i64, (y2 * 1.0e6).round() as i64),
            );
            let key_rev = (
                ((x2 * 1.0e6).round() as i64, (y2 * 1.0e6).round() as i64),
                ((x1 * 1.0e6).round() as i64, (y1 * 1.0e6).round() as i64),
            );
            assert!(
                transposed.contains(&key) || transposed.contains(&key_rev),
                "no symmetric counterpart for segment (({x1},{y1}),({x2},{y2}))"
            );
        }
    }

    #[test]
    fn axis_crossings_returns_rounded_extremes_sorted_and_deduplicated() {
        let segments: Vec<ContourSegment> =
            vec![((4.7, 50.0), (60.0, 4.7)), ((122.3, 50.0), (60.0, 122.3))];
        let crossings = axis_crossings(&segments);
        assert_eq!(crossings, vec![5, 122]);
    }

    #[test]
    fn axis_crossings_empty_input_returns_empty() {
        assert!(axis_crossings(&[]).is_empty());
    }
}

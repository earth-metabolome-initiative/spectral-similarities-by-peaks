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
use ndarray::{Array2, ArrayView2, ArrayView3, Axis};
use plotters::{
    coord::Shift,
    prelude::{
        BLACK, BitMapBackend, ChartBuilder, DrawingArea, DrawingBackend, IntoDrawingArea,
        PathElement, RGBColor, Rectangle, SVGBackend, SeriesLabelPosition, WHITE,
    },
    style::{
        Color, FontStyle, IntoFont, ShapeStyle, register_font,
        text_anchor::{HPos, Pos, VPos},
    },
};

use crate::{
    output::GridArrays,
    progress::{ProgressTask, ScanProgress},
};

/// Linear scale factor applied to every pixel dimension (canvas size,
/// margins, font sizes, stroke widths). Doubling this constant doubles the
/// rendered PNG resolution while preserving the visual layout; the SVG
/// output is also drawn larger but is unaffected visually since it stays
/// vector-based.
const RENDER_SCALE: u32 = 2;

/// Signed counterpart of `RENDER_SCALE` for the few places that take `i32`
/// (label coordinates, legend bar length).
#[allow(clippy::cast_possible_wrap)]
const RENDER_SCALE_I32: i32 = RENDER_SCALE as i32;

/// Width of each rendered heatmap image in pixels.
const IMAGE_WIDTH: u32 = 1_000 * RENDER_SCALE;
/// Height of each rendered heatmap image in pixels.
const IMAGE_HEIGHT: u32 = 900 * RENDER_SCALE;
/// Width reserved for the chart area before the colorbar.
const CHART_AREA_WIDTH: u32 = 860 * RENDER_SCALE;
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
const COLORBAR_LABEL_RIGHT_PAD: i32 = 16 * RENDER_SCALE_I32;
/// Vertical position (pixels from the top of the colorbar panel) for the
/// colorbar label's vertical center. Sits just above the colorbar top edge,
/// which is at `margin_top = 90` pixels of the same panel.
const COLORBAR_LABEL_TOP: i32 = 70 * RENDER_SCALE_I32;

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
///
/// `threshold_alphas` lists the significance levels (one curve per α) overlaid
/// on every heatmap; the same set of curves is reused across all eight metric
/// variants of one config.
pub fn write_heatmaps(
    output_dir: &Path,
    configs: &[String],
    arrays: &GridArrays,
    threshold_alphas: &[f64],
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
        let pvalue_slice = arrays
            .ks_pvalue_asymptotic
            .index_axis(Axis(0), config_index);
        // Mask every self-comparison cell (row == col) to NaN before tracing
        // any contour. The diagonal carries canonical singularities (p = 1
        // for the p-value, D = 0 for KS statistic, Δμ = 0 for mean delta)
        // that produce ring/zigzag/saddle artifacts in marching squares no
        // matter how we override them. `extract_contour` skips any quad
        // with a non-finite corner, so masking the diagonal cleanly removes
        // them from contour extraction across all three metrics.
        let mut pvalue_grid: Array2<f64> = pvalue_slice.to_owned();
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
        // KS statistic "practical effect-size" thresholds. The p-value
        // contours saturate quickly with sample size — at ~10⁵ scores per
        // cell here, even a 0.5 % CDF gap is 6σ significant — so these
        // curves answer the complementary question of where the empirical
        // CDFs actually diverge by a non-negligible amount, independent of
        // sample size. Mask the diagonal as we do for the p-value grid.
        let mut ks_statistic_grid: Array2<f64> = arrays
            .ks_statistic
            .index_axis(Axis(0), config_index)
            .to_owned();
        mask_main_diagonal(&mut ks_statistic_grid);
        for &threshold in KS_STATISTIC_THRESHOLDS {
            let segments = extract_contour(&ks_statistic_grid.view(), threshold);
            if !segments.is_empty() {
                let (x_axis_ticks, y_axis_ticks) = curve_edge_ticks(&segments);
                curves.push(ContourCurve {
                    label: format!("D = {threshold}"),
                    segments,
                    color: CONTOUR_COLORS[curves.len() % CONTOUR_COLORS.len()],
                    x_axis_ticks,
                    y_axis_ticks,
                });
            }
        }
        let overlay = ContourOverlay { curves: &curves };
        for metric in heatmap_metrics(arrays, &scales, config_index) {
            task.set_message(format!("rendering {config} {}", metric.name));
            write_heatmap_pair(&config_dir, config, &metric, &overlay, &task)?;
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

/// One contour curve rendered on top of every heatmap.
struct ContourCurve {
    /// Pre-formatted legend label for this curve.
    label: String,
    /// Unordered marching-squares segments in chart coordinates.
    segments: Vec<ContourSegment>,
    /// Stroke color used for this curve. The axis-tick labels are drawn in
    /// the same color.
    color: RGBColor,
    /// X-axis tick positions: the curve's x-coordinate at its topmost
    /// endpoint, shifted by +1 for the chart's cell-offset convention.
    /// Shown only on the x-axis (matching the value's axis).
    x_axis_ticks: Vec<i32>,
    /// Y-axis tick positions: the curve's y-coordinate at its rightmost
    /// endpoint. Shown only on the y-axis.
    y_axis_ticks: Vec<i32>,
}

/// Per-config contour set shared across all 8 metric heatmaps.
#[derive(Clone, Copy)]
struct ContourOverlay<'a> {
    /// One curve per requested significance level / effect-size threshold.
    curves: &'a [ContourCurve],
}

/// Write both SVG and PNG files for one heatmap metric.
fn write_heatmap_pair(
    output_dir: &Path,
    config: &str,
    metric: &HeatmapMetric<'_>,
    overlay: &ContourOverlay<'_>,
    progress: &ProgressTask,
) -> Result<()> {
    let stem = output_dir.join(metric.name);
    write_svg(&stem.with_extension("svg"), config, metric, overlay)?;
    progress.inc(1);
    write_png(&stem.with_extension("png"), config, metric, overlay)?;
    progress.inc(1);
    Ok(())
}

/// Write one SVG heatmap.
fn write_svg(
    path: &Path,
    config: &str,
    metric: &HeatmapMetric<'_>,
    overlay: &ContourOverlay<'_>,
) -> Result<()> {
    let root = SVGBackend::new(path, (IMAGE_WIDTH, IMAGE_HEIGHT)).into_drawing_area();
    draw_heatmap(&root, config, metric, overlay)
        .with_context(|| format!("writing SVG heatmap {}", path.display()))
}

/// Write one PNG heatmap.
fn write_png(
    path: &Path,
    config: &str,
    metric: &HeatmapMetric<'_>,
    overlay: &ContourOverlay<'_>,
) -> Result<()> {
    let root = BitMapBackend::new(path, (IMAGE_WIDTH, IMAGE_HEIGHT)).into_drawing_area();
    draw_heatmap(&root, config, metric, overlay)
        .with_context(|| format!("writing PNG heatmap {}", path.display()))
}

/// Draw one heatmap into a concrete backend.
fn draw_heatmap<Backend>(
    root: &DrawingArea<Backend, Shift>,
    config: &str,
    metric: &HeatmapMetric<'_>,
    overlay: &ContourOverlay<'_>,
) -> Result<()>
where
    Backend: DrawingBackend,
    Backend::ErrorType: std::fmt::Debug,
{
    root.fill(&WHITE).map_err(plotters_error)?;
    let (chart_area, colorbar_area) = root.split_horizontally(CHART_AREA_WIDTH);
    draw_matrix(&chart_area, config, metric, overlay)?;
    draw_colorbar(&colorbar_area, metric)?;
    root.present().map_err(plotters_error)
}

/// Draw the main matrix panel.
#[allow(clippy::too_many_lines, clippy::similar_names)]
fn draw_matrix<Backend>(
    area: &DrawingArea<Backend, Shift>,
    config: &str,
    metric: &HeatmapMetric<'_>,
    overlay: &ContourOverlay<'_>,
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
            format!("{} — {}", pretty_config_title(config), metric.title),
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
                PathElement::new(vec![(x, y), (x + 24 * RENDER_SCALE_I32, y)], style)
            });
        for points in series_iter {
            chart
                .draw_series(std::iter::once(PathElement::new(points, style)))
                .map_err(plotters_error)?;
        }
        has_legend_entry = true;
    }
    if has_legend_entry {
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
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

/// Minimum perpendicular distance (in cell units) between the curve's
/// extreme endpoint and the main diagonal for the curve to be considered
/// "converged" enough to display its asymptote tick. Curves that hug the
/// diagonal (e.g., the p-value α-contour at large samples) never separate
/// from the diagonal, so their rightmost / topmost endpoints just reflect
/// where the data array ends — labeling those would be misleading.
const ASYMPTOTE_MIN_DIAGONAL_OFFSET: f64 = 15.0;

/// Compute the per-axis tick positions for one curve. Returns
/// `(x_axis_ticks, y_axis_ticks)` where:
/// - `x_axis_ticks` is the curve's x-coordinate at its topmost endpoint
///   (where it hits the top edge of the plot), shown only on the x-axis;
/// - `y_axis_ticks` is the curve's y-coordinate at its rightmost endpoint,
///   shown only on the y-axis.
///
/// Returns empty vectors for curves that never separate sufficiently from
/// the diagonal (no horizontal/vertical asymptote to label). All positions
/// are shifted by `+1` to match the chart's cell-offset convention (sample
/// `(r, c)` is drawn at chart rect `[c+1, c+2] × [r+1, r+2]`).
fn curve_edge_ticks(segments: &[ContourSegment]) -> (Vec<i32>, Vec<i32>) {
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

/// Format a significance-level label for the heatmap legend. Uses scientific
/// notation for very small alphas.
fn format_contour_label(alpha: f64) -> String {
    if alpha < 0.001 {
        format!("α = {alpha:.2e}")
    } else {
        format!("α = {alpha}")
    }
}

/// Bright pastel stroke colors cycled through, one per significance-threshold
/// contour. Chosen to read well against both viridis and red-blue palettes.
const CONTOUR_COLORS: &[RGBColor] = &[
    RGBColor(255, 99, 132),  // pastel red
    RGBColor(102, 187, 255), // pastel sky blue
    RGBColor(120, 220, 140), // pastel green
    RGBColor(255, 180, 90),  // pastel orange
    RGBColor(190, 130, 240), // pastel violet
];

/// Sample-size-independent "practical" thresholds for the KS statistic
/// (max-CDF-gap) contour overlay, drawn in this order so the lenient curve
/// sits underneath and the strict one on top. `0.10` is the data-drift
/// literature's small/moderate boundary, `0.05` is the negligible/small
/// boundary, and `0.01` is a tighter "practically detectable" boundary
/// that hugs the diagonal where the CDFs are nearly identical.
const KS_STATISTIC_THRESHOLDS: &[f64] = &[0.10, 0.05, 0.01];

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

/// One line segment of a contour polyline, expressed in chart coordinates
/// (column-axis x, row-axis y) where the sample at `grid[(r, c)]` lives at
/// `(c + 0.5, r + 0.5)`.
type ContourSegment = ((f64, f64), (f64, f64));

/// Set every self-comparison cell (`row == col`) of `grid` to NaN, so
/// `extract_contour` skips every 2×2 quad that touches the main diagonal.
/// The diagonal carries metric-specific singularities (`p = 1`, `D = 0`,
/// `Δμ = 0`) that otherwise produce ring/zigzag/saddle artifacts.
fn mask_main_diagonal(grid: &mut Array2<f64>) {
    let len = grid.nrows().min(grid.ncols());
    for k in 0..len {
        grid[(k, k)] = f64::NAN;
    }
}

/// Trace the `alpha`-level contour of a 2D grid using marching squares with
/// linear interpolation. Returns an unordered list of line segments; segments
/// in ambiguous saddle cells are resolved against the cell centroid.
///
/// Non-finite cells short-circuit the surrounding quad (no segments emitted
/// when any of the four corners is NaN/inf), preventing spurious contours
/// through holes in the data.
fn extract_contour(grid: &ArrayView2<f64>, alpha: f64) -> Vec<ContourSegment> {
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

/// Derive the extra axis tick positions (rounded to the nearest integer)
/// where the contour comes closest to the four plot edges. Returns at most
/// four ticks (min-x, max-x, min-y, max-y), sorted and deduplicated.
///
/// Currently only used by the unit tests as the simpler reference for the
/// per-curve tick derivation in `curve_edge_ticks`.
#[allow(dead_code)]
fn axis_crossings(segments: &[ContourSegment]) -> Vec<i32> {
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
        // Bump centered at sample (10, 10), sigma^2 = 16.
        // Contour at alpha=0.5 traces a circle of radius sigma*sqrt(ln 2).
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
        // A small 5x5 grid that is symmetric under transpose; the contour
        // should mirror through the diagonal.
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

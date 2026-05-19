//! Line-plot rendering for AUROC / AUPRC pathway-discriminability series.
//!
//! Shared by the CLI plot output (`pathway_discriminability_plots/*.{svg,png}`)
//! and the WASM viewer's pathway tab so the two stay visually identical. The
//! caller provides `PathwayLineSeries` values with `(family, mz, intensity,
//! weighted)` already parsed, and this module owns the visual encoding
//! (family colour, dash pattern by m/z exponent, mix factor by intensity
//! exponent and entropy weighting) plus the chart layout and legend.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use anyhow::Result;
use plotters::{
    coord::Shift,
    element::DashedPathElement,
    prelude::{
        BLACK, ChartBuilder, DashedLineSeries, DrawingArea, DrawingBackend, IntoDrawingArea,
        LineSeries, PathElement, SVGBackend, SeriesLabelPosition, WHITE,
    },
    style::{Color, IntoFont, RGBColor, ShapeStyle},
};

use crate::{ensure_font, plotters_error};

/// Stroke width in pixels for every line drawn by this module.
const LINE_STROKE_WIDTH: u32 = 2;
/// Cosine family colour.
const COLOR_COSINE: RGBColor = RGBColor(0x1f, 0x77, 0xb4);
/// Modified cosine family colour.
const COLOR_MODIFIED_COSINE: RGBColor = RGBColor(0xff, 0x7f, 0x0e);
/// Entropy family colour.
const COLOR_ENTROPY: RGBColor = RGBColor(0x2c, 0xa0, 0x2c);
/// Modified entropy family colour.
const COLOR_MODIFIED_ENTROPY: RGBColor = RGBColor(0xd6, 0x27, 0x28);

/// Pathway-pair discriminability metric drawn on the y-axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PathwayMetric {
    /// Area under the ROC curve.
    Auroc,
    /// Area under the precision-recall curve.
    Auprc,
    /// One-vs-rest accuracy at the argmax-similarity decision rule.
    Accuracy,
    /// One-vs-rest Matthews correlation coefficient.
    Mcc,
}

impl PathwayMetric {
    /// Human-readable plot title fragment.
    #[must_use]
    pub const fn title(self) -> &'static str {
        match self {
            Self::Auroc => "Pathway-pair AUROC",
            Self::Auprc => "Pathway-pair AUPRC",
            Self::Accuracy => "Pathway-pair Accuracy",
            Self::Mcc => "Pathway-pair MCC",
        }
    }

    /// Y-axis label.
    #[must_use]
    pub const fn y_label(self) -> &'static str {
        match self {
            Self::Auroc => "AUROC",
            Self::Auprc => "AUPRC",
            Self::Accuracy => "Accuracy",
            Self::Mcc => "MCC",
        }
    }

    /// Natural value bounds of the metric. Used to clamp the auto-fit
    /// y-axis range. AUROC, AUPRC and accuracy live in `[0, 1]`. MCC lives
    /// in `[-1, 1]`.
    #[must_use]
    pub const fn value_bounds(self) -> (f64, f64) {
        match self {
            Self::Auroc | Self::Auprc | Self::Accuracy => (0.0, 1.0),
            Self::Mcc => (-1.0, 1.0),
        }
    }
}

/// Similarity-metric family encoded by line colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PathwayFamily {
    /// Direct cosine similarity.
    Cosine,
    /// Modified cosine similarity (m/z-aware peak alignment).
    ModifiedCosine,
    /// Direct entropy similarity.
    Entropy,
    /// Modified entropy similarity (m/z-aware peak alignment).
    ModifiedEntropy,
}

impl PathwayFamily {
    /// Stable rank used to group the legend by family.
    const fn rank(self) -> u8 {
        match self {
            Self::Cosine => 0,
            Self::ModifiedCosine => 1,
            Self::Entropy => 2,
            Self::ModifiedEntropy => 3,
        }
    }

    /// Base colour for this family.
    const fn color(self) -> RGBColor {
        match self {
            Self::Cosine => COLOR_COSINE,
            Self::ModifiedCosine => COLOR_MODIFIED_COSINE,
            Self::Entropy => COLOR_ENTROPY,
            Self::ModifiedEntropy => COLOR_MODIFIED_ENTROPY,
        }
    }

    /// Human-readable family name, with the entropy-weighting flag folded
    /// in. Cosine and modified-cosine variants carry no weighting concept,
    /// so the `weighted` argument is ignored for them.
    #[must_use]
    pub const fn pretty_name(self, weighted: Option<bool>) -> &'static str {
        match (self, weighted) {
            (Self::Cosine, _) => "Cosine",
            (Self::ModifiedCosine, _) => "Modified Cosine",
            (Self::Entropy, Some(true)) => "Weighted Entropy",
            (Self::Entropy, _) => "Entropy",
            (Self::ModifiedEntropy, Some(true)) => "Weighted Modified Entropy",
            (Self::ModifiedEntropy, _) => "Modified Entropy",
        }
    }
}

/// Human-readable legend label, shape `Family[, weighted], m/z=…, int=…`.
///
/// Both exponents are always emitted so configs that differ only by
/// exponent stay disambiguated.
#[must_use]
pub fn pretty_series_label(
    family: PathwayFamily,
    mz_exp: f64,
    intensity_exp: f64,
    weighted: Option<bool>,
) -> String {
    format!(
        "{}, m/z={mz_exp:.1}, int={intensity_exp:.2}",
        family.pretty_name(weighted)
    )
}

/// One line drawn on the chart. The caller pre-parses the slug into
/// `(family, mz, intensity, weighted)`. `points` is `(peak_count, value)`
/// for the metric currently being plotted.
#[derive(Clone, Debug)]
pub struct PathwayLineSeries {
    /// Legend label, typically the config slug.
    pub label: String,
    /// Similarity-metric family, determines the base colour.
    pub family: PathwayFamily,
    /// m/z exponent, determines the dash pattern.
    pub mz_exp: f64,
    /// Intensity exponent, modulates the colour mix factor.
    pub intensity_exp: f64,
    /// Optional entropy weighting flag. `Some(false)` darkens the colour.
    pub weighted: Option<bool>,
    /// Data points as `(peak_count, metric_value)`. Non-finite values are
    /// dropped before plotting.
    pub points: Vec<(i32, f64)>,
}

/// Render the chart to an in-memory SVG string. Used by the WASM viewer to
/// inject the result via `dangerous_inner_html`.
///
/// # Errors
///
/// Returns an error when the plotters backend fails to produce a chart
/// (out-of-memory or invalid layout coordinates).
pub fn render_pathway_lines_svg(
    title: &str,
    metric: PathwayMetric,
    series: &[PathwayLineSeries],
    width: u32,
    height: u32,
) -> Result<String> {
    ensure_font()?;
    let mut buffer = String::new();
    {
        let root = SVGBackend::with_string(&mut buffer, (width, height)).into_drawing_area();
        draw_pathway_lines(&root, title, metric, series)?;
    }
    Ok(buffer)
}

/// Draw the chart onto an existing plotters drawing area. Used by the CLI
/// to share the body across `SVGBackend` (file) and `BitMapBackend` (PNG)
/// outputs.
///
/// # Errors
///
/// Returns an error when plotters fails to lay out or draw the chart.
pub fn draw_pathway_lines<Backend>(
    root: &DrawingArea<Backend, Shift>,
    title: &str,
    metric: PathwayMetric,
    series: &[PathwayLineSeries],
) -> Result<()>
where
    Backend: DrawingBackend,
    Backend::ErrorType: std::fmt::Debug,
{
    ensure_font()?;
    root.fill(&WHITE).map_err(plotters_error)?;
    let (y_min, y_max) = focused_y_range_for_metric(
        series.iter().flat_map(|s| s.points.iter().map(|(_, v)| *v)),
        metric,
    );
    let x_end = largest_peak_count(series).saturating_add(1).max(2);
    let mut chart = ChartBuilder::on(root)
        .caption(title, ("sans-serif", 26))
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

    let mut indexed: Vec<&PathwayLineSeries> = series.iter().collect();
    indexed.sort_by(|a, b| {
        a.family
            .rank()
            .cmp(&b.family.rank())
            .then_with(|| float_cmp(a.mz_exp, b.mz_exp))
            .then_with(|| float_cmp(b.intensity_exp, a.intensity_exp))
            .then_with(|| weighted_rank(a.weighted).cmp(&weighted_rank(b.weighted)))
            .then_with(|| a.label.cmp(&b.label))
    });

    for entry in indexed {
        let style = series_style(
            entry.family,
            entry.mz_exp,
            entry.intensity_exp,
            entry.weighted,
        );
        let mut points: Vec<(i32, f64)> = entry
            .points
            .iter()
            .filter(|(_, v)| v.is_finite())
            .copied()
            .collect();
        points.sort_by_key(|&(x, _)| x);
        if points.is_empty() {
            continue;
        }
        let shape = style.shape;
        let label = entry.label.clone();
        match style.dash {
            None => {
                chart
                    .draw_series(LineSeries::new(points, shape))
                    .map_err(plotters_error)?
                    .label(label)
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 28, y)], shape));
            }
            Some((size, spacing)) => {
                chart
                    .draw_series(DashedLineSeries::new(points, size, spacing, shape))
                    .map_err(plotters_error)?
                    .label(label)
                    .legend(move |(x, y)| {
                        DashedPathElement::new(vec![(x, y), (x + 28, y)], size, spacing, shape)
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

/// Concrete drawing style for one series.
#[derive(Clone, Copy)]
struct SeriesStyle {
    /// Stroke colour and width.
    shape: ShapeStyle,
    /// Optional `(dash_size, gap_size)` in pixels. `None` draws a solid line.
    dash: Option<(u32, u32)>,
}

/// Build the concrete drawing style (colour, stroke width and optional dash
/// pattern) from the four style axes parsed off the config slug.
fn series_style(
    family: PathwayFamily,
    mz: f64,
    intensity: f64,
    weighted: Option<bool>,
) -> SeriesStyle {
    let dash = if (mz - 0.0).abs() < 0.5 {
        None
    } else if (mz - 1.0).abs() < 0.5 {
        Some((10, 6))
    } else {
        Some((3, 5))
    };
    let intensity_factor = if (intensity - 1.0).abs() < 0.05 {
        1.0
    } else if intensity >= 0.5 {
        0.78
    } else {
        0.5
    };
    let weighted_factor = match weighted {
        Some(false) => 0.62,
        _ => 1.0,
    };
    let mix = intensity_factor * weighted_factor;
    let base = family.color();
    let shape = if (mix - 1.0_f64).abs() < 0.01 {
        base.stroke_width(LINE_STROKE_WIDTH)
    } else {
        base.mix(mix).stroke_width(LINE_STROKE_WIDTH)
    };
    SeriesStyle { shape, dash }
}

/// Compute a focused y-axis range from finite values.
///
/// Pads the observed min and max by 8 % of the span so curves don't touch
/// the chart frame. The result is clipped to the metric's natural value
/// bounds (`[0, 1]` for AUROC / AUPRC / accuracy, `[-1, 1]` for MCC).
#[must_use]
pub fn focused_y_range_for_metric(
    values: impl IntoIterator<Item = f64>,
    metric: PathwayMetric,
) -> (f64, f64) {
    let (bound_lo, bound_hi) = metric.value_bounds();
    let (min, max) = values
        .into_iter()
        .filter(|v| v.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        });
    if !min.is_finite() || !max.is_finite() {
        return (bound_lo, bound_hi);
    }
    let span = (max - min).max(0.01);
    let pad = span * 0.08;
    let lower = (min - pad).max(bound_lo);
    let upper = (max + pad).min(bound_hi);
    if upper - lower < 0.01 {
        ((lower - 0.005).max(bound_lo), (upper + 0.005).min(bound_hi))
    } else {
        (lower, upper)
    }
}

/// Largest `x` coordinate (peak count) seen across every series, with a
/// sensible default when the input is empty.
fn largest_peak_count(series: &[PathwayLineSeries]) -> i32 {
    series
        .iter()
        .flat_map(|s| s.points.iter().map(|(x, _)| *x))
        .max()
        .unwrap_or(128)
}

/// Compare two `f64`s for sort ordering, treating `NaN` as equal so the
/// caller's sort never panics on malformed input.
fn float_cmp(a: f64, b: f64) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
}

/// Stable ordering for the optional `weighted` flag. `None` comes first,
/// then `Some(true)`, then `Some(false)`.
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
    fn focused_range_handles_empty_input() {
        assert_eq!(
            focused_y_range_for_metric(std::iter::empty(), PathwayMetric::Auroc),
            (0.0, 1.0)
        );
    }

    #[test]
    fn focused_range_pads_observed_values() {
        let (lo, hi) = focused_y_range_for_metric([0.50, 0.60, 0.55], PathwayMetric::Auroc);
        assert!(lo < 0.50);
        assert!(hi > 0.60);
        assert!(lo >= 0.0);
        assert!(hi <= 1.0);
    }

    #[test]
    fn focused_range_clamps_to_unit_interval() {
        let (lo, hi) = focused_y_range_for_metric([0.0, 1.0], PathwayMetric::Auroc);
        assert!(lo.abs() < 1e-12);
        assert!((hi - 1.0).abs() < 1e-12);
    }

    #[test]
    fn focused_range_clamps_mcc_to_signed_interval() {
        let (lo, hi) = focused_y_range_for_metric([-1.0, 1.0], PathwayMetric::Mcc);
        assert!((lo + 1.0).abs() < 1e-12);
        assert!((hi - 1.0).abs() < 1e-12);
    }

    #[test]
    fn render_lines_svg_emits_svg_root() -> Result<()> {
        let series = vec![
            PathwayLineSeries {
                label: "cosine_mz0.000_int1.000".into(),
                family: PathwayFamily::Cosine,
                mz_exp: 0.0,
                intensity_exp: 1.0,
                weighted: None,
                points: vec![(1, 0.51), (16, 0.55), (128, 0.50)],
            },
            PathwayLineSeries {
                label: "modified_cosine_mz1.000_int0.500".into(),
                family: PathwayFamily::ModifiedCosine,
                mz_exp: 1.0,
                intensity_exp: 0.5,
                weighted: None,
                points: vec![(1, 0.49), (16, 0.53), (128, 0.49)],
            },
        ];
        let svg = render_pathway_lines_svg(
            "Pathway-pair AUROC (test)",
            PathwayMetric::Auroc,
            &series,
            640,
            420,
        )?;
        assert!(svg.contains("<svg"));
        assert!(svg.contains("cosine_mz0.000_int1.000"));
        Ok(())
    }
}

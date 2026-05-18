//! File-I/O wrappers around the [`spectral_render`] heatmap pipeline.
//!
//! The pure plotters-based rendering lives in the `spectral-render` crate so
//! it can be shared with the browser viewer. This module keeps only the
//! pieces that need filesystem access: the per-config orchestration loop
//! over [`crate::output::GridArrays`], SVG and PNG file writers, and a few helpers
//! re-exported for other CLI modules.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use anyhow::{Context, Result, bail};
use plotters::prelude::{BitMapBackend, IntoDrawingArea, SVGBackend};
use plotters::style::{FontStyle, register_font};
use spectral_render::{
    ContourOverlay, GridViews, HeatmapMetric, IMAGE_HEIGHT, IMAGE_WIDTH, KS_STATISTIC_THRESHOLDS,
    Metric, Scales, build_contour_curves, draw_heatmap, ensure_font, metric_view,
};

pub use spectral_render::{ks_statistic_asymptote, plotters_error};

use crate::{
    output::GridArrays,
    progress::{ProgressTask, ScanProgress},
};

/// Default per-config significance levels for p-value contour overlays.
const DEFAULT_CONTOUR_ALPHAS: &[f64] = &[0.05];

/// Environment variable that overrides the embedded font with one read from disk.
///
/// Primarily exists so tests can inject a font failure deterministically. When
/// unset, the embedded font shipped in `spectral-render` is used.
const HEATMAP_FONT_ENV_VAR: &str = "SPECTRAL_SIMILARITIES_FONT";

/// Cached result of resolving the heatmap font choice for this process.
static HEATMAP_FONT_REGISTRATION: OnceLock<std::result::Result<(), String>> = OnceLock::new();

/// Register the heatmap font with plotters, preferring an override from
/// [`HEATMAP_FONT_ENV_VAR`] when set and otherwise falling back to the
/// embedded font in `spectral-render`.
///
/// # Errors
///
/// Returns an error when the override path is set but the referenced file
/// cannot be read or registered. The embedded fallback cannot fail.
pub fn ensure_heatmap_font() -> Result<()> {
    match HEATMAP_FONT_REGISTRATION.get_or_init(register_heatmap_font) {
        Ok(()) => Ok(()),
        Err(message) => bail!("{message}"),
    }
}

/// Resolve which font bytes to feed plotters. Honors the env-var override
/// when set; otherwise delegates to the embedded font registration.
fn register_heatmap_font() -> std::result::Result<(), String> {
    if let Some(path) = env::var_os(HEATMAP_FONT_ENV_VAR) {
        let path = PathBuf::from(path);
        let bytes = fs::read(&path).map_err(|error| {
            format!(
                "failed to register font from {HEATMAP_FONT_ENV_VAR}={}: failed to read {}: {error}",
                path.display(),
                path.display()
            )
        })?;
        let leaked = Box::leak(bytes.into_boxed_slice());
        return register_font("sans-serif", FontStyle::Normal, leaked)
            .map_err(|_| format!("{} is not a valid TrueType/OpenType font", path.display()));
    }
    ensure_font().map_err(|error| error.to_string())
}

/// Write SVG and PNG heatmaps for each dense grid matrix and config.
///
/// `threshold_alphas` lists the significance levels (one curve per α) overlaid
/// on every heatmap; the same set of curves is reused across all eight metric
/// variants of one config.
///
/// # Errors
///
/// Returns an error when output directories cannot be created, the font
/// cannot be registered, or any rendering call fails.
pub fn write_heatmaps(
    output_dir: &Path,
    configs: &[String],
    arrays: &GridArrays,
    threshold_alphas: &[f64],
    progress: &ScanProgress,
) -> Result<()> {
    validate_config_axis(configs, arrays)?;
    ensure_heatmap_font()?;
    let dataset_label = output_dir
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string);
    let output_dir = output_dir.join("heatmaps");
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("creating {}", output_dir.display()))?;

    let grids = GridViews {
        mean_delta: arrays.mean_delta.view(),
        ks_statistic: arrays.ks_statistic.view(),
        ks_pvalue_asymptotic: arrays.ks_pvalue_asymptotic.view(),
        wasserstein_1d: arrays.wasserstein_1d.view(),
    };
    let scales = Scales::from_grids(&grids);
    let total_heatmaps = configs
        .len()
        .checked_mul(Metric::ALL.len())
        .and_then(|count| count.checked_mul(2))
        .unwrap_or(usize::MAX);
    let task = progress.bar(
        u64::try_from(total_heatmaps).unwrap_or(u64::MAX),
        "rendering heatmaps",
    );
    let alphas = if threshold_alphas.is_empty() {
        DEFAULT_CONTOUR_ALPHAS
    } else {
        threshold_alphas
    };
    for (config_index, config) in configs.iter().enumerate() {
        let config_dir = output_dir.join(sanitize_path_component(config));
        fs::create_dir_all(&config_dir)
            .with_context(|| format!("creating {}", config_dir.display()))?;
        let curves = build_contour_curves(&grids, config_index, alphas, KS_STATISTIC_THRESHOLDS);
        let overlay = ContourOverlay { curves: &curves };
        for metric in Metric::ALL {
            let view = metric_view(metric, grids, &scales, config_index);
            task.set_message(format!("rendering {config} {}", view.name));
            write_heatmap_pair(
                &config_dir,
                config,
                &view,
                &overlay,
                dataset_label.as_deref(),
                &task,
            )?;
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

/// Write both SVG and PNG files for one heatmap metric.
fn write_heatmap_pair(
    output_dir: &Path,
    config: &str,
    metric: &HeatmapMetric<'_>,
    overlay: &ContourOverlay<'_>,
    dataset_label: Option<&str>,
    progress: &ProgressTask,
) -> Result<()> {
    let stem = output_dir.join(metric.name);
    write_svg(
        &stem.with_extension("svg"),
        config,
        metric,
        overlay,
        dataset_label,
    )?;
    progress.inc(1);
    write_png(
        &stem.with_extension("png"),
        config,
        metric,
        overlay,
        dataset_label,
    )?;
    progress.inc(1);
    Ok(())
}

/// Write one SVG heatmap.
fn write_svg(
    path: &Path,
    config: &str,
    metric: &HeatmapMetric<'_>,
    overlay: &ContourOverlay<'_>,
    dataset_label: Option<&str>,
) -> Result<()> {
    let root = SVGBackend::new(path, (IMAGE_WIDTH, IMAGE_HEIGHT)).into_drawing_area();
    draw_heatmap(&root, config, metric, overlay, dataset_label)
        .with_context(|| format!("writing SVG heatmap {}", path.display()))
}

/// Write one PNG heatmap.
fn write_png(
    path: &Path,
    config: &str,
    metric: &HeatmapMetric<'_>,
    overlay: &ContourOverlay<'_>,
    dataset_label: Option<&str>,
) -> Result<()> {
    let root = BitMapBackend::new(path, (IMAGE_WIDTH, IMAGE_HEIGHT)).into_drawing_area();
    draw_heatmap(&root, config, metric, overlay, dataset_label)
        .with_context(|| format!("writing PNG heatmap {}", path.display()))
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

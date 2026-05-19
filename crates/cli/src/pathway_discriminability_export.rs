//! JSON export of pathway-discriminability parquets for the WASM viewer.
//!
//! The viewer needs to pick a pathway, choose a metric (AUROC or AUPRC),
//! and toggle similarity-family / m/z / intensity / weighted filters
//! interactively. Shipping the two parquet files to the browser is heavy
//! (we would need a WASM parquet reader); instead this module produces a
//! single compact JSON document containing the same numbers, structured
//! per pathway so the viewer can render a chart from one in-memory parse.
//!
//! Layout:
//!
//! ```json
//! {
//!   "peak_counts": [1, 2, ..., 128],
//!   "configs": [
//!     {"slug": "cosine_mz0.000_int1.000", "family": "cosine",
//!      "mz_exp": 0.0, "intensity_exp": 1.0, "weighted": null},
//!     ...
//!   ],
//!   "pathways": [
//!     {"label": "Aggregate (micro-averaged)", "kind": "aggregate",
//!      "auroc": [[<peak_counts.len() floats or null>], ...configs.len()],
//!      "auprc": [...],
//!      "accuracy": null,
//!      "mcc": null},
//!     {"label": "Alkaloids", "kind": "per_class",
//!      "auroc": [...], "auprc": [...], "accuracy": [...], "mcc": [...]},
//!     ... 6 more base pathways ...
//!     {"label": "All pathways (support-weighted)", "kind": "aggregate_weighted",
//!      "auroc": null, "auprc": null,
//!      "accuracy": [...], "mcc": [...]}
//!   ]
//! }
//! ```
//!
//! Pipe-separated multi-pathway labels in the per-class parquet have NaN
//! AUROC and AUPRC by construction (zero positives once the candidate
//! pathway must match exactly), so they are skipped. The accuracy / MCC
//! matrices come from `pathway_prediction_metrics.parquet`, which uses
//! the same `(config, peak_count, pathway)` key and adds a sentinel
//! `weighted_average` pathway re-exposed as `"All pathways
//! (support-weighted)"`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::BufWriter,
    path::Path,
};

use anyhow::{Context, Result};
use arrow_array::{Array, BooleanArray, Float64Array, RecordBatch, StringArray, UInt64Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Serialize;
use spectral_render::PathwayFamily;

use crate::{
    pathway_discriminability_plots::{ParsedConfig, parse_config_slug},
    progress::ScanProgress,
};

/// File name written under `output_dir`.
pub const OUTPUT_FILE_NAME: &str = "pathway_discriminability_lines.json";

/// `(config, peak_count)` grid of optional values, indexed as `[config][peak]`.
type MetricMatrix = Vec<Vec<Option<f64>>>;

/// Read the aggregate and per-class discriminability parquets in
/// `output_dir`, build the compact JSON the WASM viewer needs, and write
/// it to `output_dir/pathway_discriminability_lines.json`. Returns `Ok(())`
/// without writing the file when neither parquet exists.
///
/// # Errors
///
/// Returns an error if either parquet exists but cannot be read, or if the
/// JSON file cannot be written to disk.
pub fn export_pathway_discriminability_json(
    output_dir: &Path,
    progress: &ScanProgress,
) -> Result<()> {
    let aggregate_path = output_dir.join("pathway_discriminability.parquet");
    let per_class_path = output_dir.join("pathway_discriminability_per_class.parquet");
    let prediction_metrics_path = output_dir.join("pathway_prediction_metrics.parquet");

    let aggregate_rows = if aggregate_path.is_file() {
        let spinner = progress.spinner("reading pathway_discriminability.parquet");
        let rows = read_aggregate_rows(&aggregate_path)?;
        spinner.finish();
        rows
    } else {
        Vec::new()
    };
    let per_class_rows = if per_class_path.is_file() {
        let spinner = progress.spinner("reading pathway_discriminability_per_class.parquet");
        let rows = read_per_class_rows(&per_class_path)?;
        spinner.finish();
        rows
    } else {
        Vec::new()
    };
    let prediction_metric_rows = if prediction_metrics_path.is_file() {
        let spinner = progress.spinner("reading pathway_prediction_metrics.parquet");
        let rows = read_prediction_metric_rows(&prediction_metrics_path)?;
        spinner.finish();
        rows
    } else {
        Vec::new()
    };

    if aggregate_rows.is_empty() && per_class_rows.is_empty() && prediction_metric_rows.is_empty() {
        return Ok(());
    }

    let document = build_document(&aggregate_rows, &per_class_rows, &prediction_metric_rows);
    if document.pathways.is_empty() {
        return Ok(());
    }
    let target = output_dir.join(OUTPUT_FILE_NAME);
    let write_progress = progress.spinner(format!("writing {}", target.display()));
    let file =
        fs::File::create(&target).with_context(|| format!("creating {}", target.display()))?;
    serde_json::to_writer(BufWriter::new(file), &document)
        .with_context(|| format!("writing {}", target.display()))?;
    write_progress.finish();
    Ok(())
}

/// One row of `pathway_discriminability.parquet`.
struct AggregateRow {
    /// Similarity-config slug.
    config: String,
    /// Number of retained top-intensity peaks.
    peak_count: u64,
    /// Pooled AUROC across every `(query, candidate)` pair.
    auroc: Option<f64>,
    /// Pooled AUPRC across every `(query, candidate)` pair.
    auprc: Option<f64>,
}

/// One row of `pathway_discriminability_per_class.parquet`.
struct PerClassRow {
    /// Similarity-config slug.
    config: String,
    /// Number of retained top-intensity peaks.
    peak_count: u64,
    /// Positive-class pathway label.
    pathway: String,
    /// One-vs-rest AUROC for the named pathway.
    auroc: Option<f64>,
    /// One-vs-rest AUPRC for the named pathway.
    auprc: Option<f64>,
}

/// One row of `pathway_prediction_metrics.parquet`.
struct PredictionMetricRow {
    /// Similarity-config slug.
    config: String,
    /// Number of retained top-intensity peaks.
    peak_count: u64,
    /// Pathway label (or the `weighted_average` sentinel).
    pathway: String,
    /// True when the row holds the support-weighted average across pathways.
    is_weighted_average: bool,
    /// One-vs-rest accuracy.
    accuracy: Option<f64>,
    /// One-vs-rest Matthews correlation coefficient.
    mcc: Option<f64>,
}

/// Top-level JSON object written to disk.
#[derive(Serialize)]
struct Document {
    /// Sorted list of peak counts present in the data, used as the
    /// x-coordinate axis.
    peak_counts: Vec<u64>,
    /// Sorted list of similarity configs present in the data.
    configs: Vec<ConfigEntry>,
    /// One entry per pathway (plus an aggregate row when available).
    pathways: Vec<PathwayEntry>,
}

/// Pre-parsed config slug plus its visual-encoding axes.
#[derive(Serialize)]
struct ConfigEntry {
    /// Raw config slug used as the legend label.
    slug: String,
    /// Family identifier, lowercase string.
    family: &'static str,
    /// m/z exponent for peak weighting.
    mz_exp: f64,
    /// Intensity exponent for peak weighting.
    intensity_exp: f64,
    /// Optional entropy-weighting flag.
    weighted: Option<bool>,
}

/// One pathway's AUROC / AUPRC / accuracy / MCC matrices. Any metric that
/// is not defined for the entry is serialised as JSON `null`.
#[derive(Serialize)]
struct PathwayEntry {
    /// Display label.
    label: String,
    /// One of `"aggregate"`, `"per_class"`, `"aggregate_weighted"`.
    kind: &'static str,
    /// `configs.len()` rows by `peak_counts.len()` columns of AUROC.
    auroc: Option<Vec<Vec<Option<f64>>>>,
    /// `configs.len()` rows by `peak_counts.len()` columns of AUPRC.
    auprc: Option<Vec<Vec<Option<f64>>>>,
    /// `configs.len()` rows by `peak_counts.len()` columns of accuracy.
    accuracy: Option<Vec<Vec<Option<f64>>>>,
    /// `configs.len()` rows by `peak_counts.len()` columns of MCC.
    mcc: Option<Vec<Vec<Option<f64>>>>,
}

/// Convert the parsed `PathwayFamily` to the lowercase string used by the
/// JSON `family` field.
const fn family_str(family: PathwayFamily) -> &'static str {
    match family {
        PathwayFamily::Cosine => "cosine",
        PathwayFamily::ModifiedCosine => "modified-cosine",
        PathwayFamily::Entropy => "entropy",
        PathwayFamily::ModifiedEntropy => "modified-entropy",
    }
}

/// Build the JSON document from the aggregate, per-class and
/// prediction-metric rows.
fn build_document(
    aggregate_rows: &[AggregateRow],
    per_class_rows: &[PerClassRow],
    prediction_rows: &[PredictionMetricRow],
) -> Document {
    let (peak_counts, config_slugs) = collect_axes(aggregate_rows, per_class_rows, prediction_rows);
    let configs: Vec<ConfigEntry> = config_slugs
        .iter()
        .map(String::as_str)
        .map(config_entry_for)
        .collect();

    let prediction_by_pathway = group_predictions(prediction_rows, false);
    let weighted_average_rows: Vec<&PredictionMetricRow> = prediction_rows
        .iter()
        .filter(|row| row.is_weighted_average)
        .collect();
    let per_class_by_pathway = group_per_class(per_class_rows);

    let mut pathways: Vec<PathwayEntry> = Vec::new();
    if let Some(entry) = aggregate_entry(aggregate_rows, &peak_counts, &config_slugs) {
        pathways.push(entry);
    }
    let mut covered_pathways: BTreeSet<String> = BTreeSet::new();
    for (label, rows) in &per_class_by_pathway {
        covered_pathways.insert(label.clone());
        pathways.push(per_class_entry(
            label,
            rows,
            prediction_by_pathway.get(label),
            &peak_counts,
            &config_slugs,
        ));
    }
    // Surface any pathway that has prediction-metric data but no AUROC /
    // AUPRC row (unlikely in practice but keeps the surface stable).
    for (label, rows) in &prediction_by_pathway {
        if covered_pathways.contains(label) {
            continue;
        }
        let (accuracy, mcc) =
            build_prediction_matrix(&peak_counts, &config_slugs, rows.iter().copied());
        pathways.push(PathwayEntry {
            label: label.clone(),
            kind: "per_class",
            auroc: None,
            auprc: None,
            accuracy: Some(accuracy),
            mcc: Some(mcc),
        });
    }
    if let Some(entry) = weighted_average_entry(&weighted_average_rows, &peak_counts, &config_slugs)
    {
        pathways.push(entry);
    }

    Document {
        peak_counts,
        configs,
        pathways,
    }
}

/// Collect the sorted union of peak counts and config slugs across the
/// three input parquets.
fn collect_axes(
    aggregate_rows: &[AggregateRow],
    per_class_rows: &[PerClassRow],
    prediction_rows: &[PredictionMetricRow],
) -> (Vec<u64>, Vec<String>) {
    let mut peak_counts: BTreeSet<u64> = BTreeSet::new();
    let mut config_slugs: BTreeSet<String> = BTreeSet::new();
    for row in aggregate_rows {
        peak_counts.insert(row.peak_count);
        config_slugs.insert(row.config.clone());
    }
    for row in per_class_rows {
        peak_counts.insert(row.peak_count);
        config_slugs.insert(row.config.clone());
    }
    for row in prediction_rows {
        peak_counts.insert(row.peak_count);
        config_slugs.insert(row.config.clone());
    }
    (
        peak_counts.into_iter().collect(),
        config_slugs.into_iter().collect(),
    )
}

/// Bucket per-class rows by pathway, dropping pipe-separated combos.
fn group_per_class(per_class_rows: &[PerClassRow]) -> BTreeMap<String, Vec<&PerClassRow>> {
    let mut grouped: BTreeMap<String, Vec<&PerClassRow>> = BTreeMap::new();
    for row in per_class_rows {
        if row.pathway.contains('|') {
            continue;
        }
        grouped.entry(row.pathway.clone()).or_default().push(row);
    }
    grouped
}

/// Build a JSON `ConfigEntry` from one parsed slug.
fn config_entry_for(slug: &str) -> ConfigEntry {
    let parsed: ParsedConfig = parse_config_slug(slug);
    ConfigEntry {
        slug: slug.to_string(),
        family: family_str(parsed.family),
        mz_exp: parsed.mz,
        intensity_exp: parsed.intensity,
        weighted: parsed.weighted,
    }
}

/// Build the aggregate (micro-averaged) pathway entry, if any aggregate
/// rows were loaded.
fn aggregate_entry(
    aggregate_rows: &[AggregateRow],
    peak_counts: &[u64],
    config_slugs: &[String],
) -> Option<PathwayEntry> {
    if aggregate_rows.is_empty() {
        return None;
    }
    let (auroc, auprc) = build_auc_matrix(
        peak_counts,
        config_slugs,
        aggregate_rows
            .iter()
            .map(|row| (row.config.as_str(), row.peak_count, row.auroc, row.auprc)),
    );
    Some(PathwayEntry {
        label: "Aggregate (micro-averaged)".to_string(),
        kind: "aggregate",
        auroc: Some(auroc),
        auprc: Some(auprc),
        accuracy: None,
        mcc: None,
    })
}

/// Build a per-class pathway entry, optionally enriched with accuracy /
/// MCC pulled from the prediction-metric bucket for the same label.
fn per_class_entry(
    label: &str,
    rows: &[&PerClassRow],
    predictions: Option<&Vec<&PredictionMetricRow>>,
    peak_counts: &[u64],
    config_slugs: &[String],
) -> PathwayEntry {
    let (auroc, auprc) = build_auc_matrix(
        peak_counts,
        config_slugs,
        rows.iter()
            .map(|row| (row.config.as_str(), row.peak_count, row.auroc, row.auprc)),
    );
    let (accuracy, mcc) = predictions
        .map(|rows| build_prediction_matrix(peak_counts, config_slugs, rows.iter().copied()))
        .map_or((None, None), |(a, m)| (Some(a), Some(m)));
    PathwayEntry {
        label: label.to_string(),
        kind: "per_class",
        auroc: Some(auroc),
        auprc: Some(auprc),
        accuracy,
        mcc,
    }
}

/// Build the "All pathways (support-weighted)" entry, if any
/// `weighted_average` sentinel rows were loaded.
fn weighted_average_entry(
    weighted_average_rows: &[&PredictionMetricRow],
    peak_counts: &[u64],
    config_slugs: &[String],
) -> Option<PathwayEntry> {
    if weighted_average_rows.is_empty() {
        return None;
    }
    let (accuracy, mcc) = build_prediction_matrix(
        peak_counts,
        config_slugs,
        weighted_average_rows.iter().copied(),
    );
    Some(PathwayEntry {
        label: "All pathways (support-weighted)".to_string(),
        kind: "aggregate_weighted",
        auroc: None,
        auprc: None,
        accuracy: Some(accuracy),
        mcc: Some(mcc),
    })
}

/// Materialise the AUROC / AUPRC matrices for one pathway.
fn build_auc_matrix<'a, I>(
    peak_counts: &[u64],
    config_slugs: &[String],
    rows: I,
) -> (MetricMatrix, MetricMatrix)
where
    I: IntoIterator<Item = (&'a str, u64, Option<f64>, Option<f64>)>,
{
    let config_index: BTreeMap<&str, usize> = config_slugs
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let peak_index: BTreeMap<u64, usize> = peak_counts
        .iter()
        .enumerate()
        .map(|(i, p)| (*p, i))
        .collect();
    let mut auroc = vec![vec![None; peak_counts.len()]; config_slugs.len()];
    let mut auprc = vec![vec![None; peak_counts.len()]; config_slugs.len()];
    for (config, peak, a, b) in rows {
        let Some(&ci) = config_index.get(config) else {
            continue;
        };
        let Some(&pi) = peak_index.get(&peak) else {
            continue;
        };
        auroc[ci][pi] = a.filter(|v| v.is_finite());
        auprc[ci][pi] = b.filter(|v| v.is_finite());
    }
    (auroc, auprc)
}

/// Materialise the accuracy / MCC matrices for one pathway from a slice
/// of prediction-metric rows.
fn build_prediction_matrix<'a, I>(
    peak_counts: &[u64],
    config_slugs: &[String],
    rows: I,
) -> (MetricMatrix, MetricMatrix)
where
    I: IntoIterator<Item = &'a PredictionMetricRow>,
{
    let config_index: BTreeMap<&str, usize> = config_slugs
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let peak_index: BTreeMap<u64, usize> = peak_counts
        .iter()
        .enumerate()
        .map(|(i, p)| (*p, i))
        .collect();
    let mut accuracy = vec![vec![None; peak_counts.len()]; config_slugs.len()];
    let mut mcc = vec![vec![None; peak_counts.len()]; config_slugs.len()];
    for row in rows {
        let Some(&ci) = config_index.get(row.config.as_str()) else {
            continue;
        };
        let Some(&pi) = peak_index.get(&row.peak_count) else {
            continue;
        };
        accuracy[ci][pi] = row.accuracy.filter(|v| v.is_finite());
        mcc[ci][pi] = row.mcc.filter(|v| v.is_finite());
    }
    (accuracy, mcc)
}

/// Bucket the prediction-metric rows by pathway label, optionally keeping
/// the `weighted_average` sentinel rows in their own bucket.
fn group_predictions(
    rows: &[PredictionMetricRow],
    keep_weighted_average: bool,
) -> BTreeMap<String, Vec<&PredictionMetricRow>> {
    let mut grouped: BTreeMap<String, Vec<&PredictionMetricRow>> = BTreeMap::new();
    for row in rows {
        if row.is_weighted_average != keep_weighted_average {
            continue;
        }
        if row.pathway.contains('|') {
            continue;
        }
        grouped.entry(row.pathway.clone()).or_default().push(row);
    }
    grouped
}

/// Read every row of `pathway_discriminability.parquet`.
fn read_aggregate_rows(path: &Path) -> Result<Vec<AggregateRow>> {
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

/// Read every row of `pathway_prediction_metrics.parquet`.
fn read_prediction_metric_rows(path: &Path) -> Result<Vec<PredictionMetricRow>> {
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading metadata from {}", path.display()))?
        .build()
        .with_context(|| format!("building reader for {}", path.display()))?;
    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch.with_context(|| format!("decoding batch in {}", path.display()))?;
        observe_prediction_batch(&batch, &mut rows)?;
    }
    Ok(rows)
}

/// Extract every prediction-metric row from one record batch.
fn observe_prediction_batch(
    batch: &RecordBatch,
    rows: &mut Vec<PredictionMetricRow>,
) -> Result<()> {
    let configs = required_column::<StringArray>(batch, "config")?;
    let peak_counts = required_column::<UInt64Array>(batch, "peak_count")?;
    let pathways = required_column::<StringArray>(batch, "pathway")?;
    let is_weighted = required_column::<BooleanArray>(batch, "is_weighted_average")?;
    let accuracy = required_column::<Float64Array>(batch, "accuracy")?;
    let mcc = required_column::<Float64Array>(batch, "mcc")?;
    for row in 0..batch.num_rows() {
        rows.push(PredictionMetricRow {
            config: configs.value(row).to_string(),
            peak_count: peak_counts.value(row),
            pathway: pathways.value(row).to_string(),
            is_weighted_average: is_weighted.value(row),
            accuracy: cell_or_none(accuracy, row),
            mcc: cell_or_none(mcc, row),
        });
    }
    Ok(())
}

/// Read every row of `pathway_discriminability_per_class.parquet`.
fn read_per_class_rows(path: &Path) -> Result<Vec<PerClassRow>> {
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

/// Extract every aggregate row from one record batch.
fn observe_aggregate_batch(batch: &RecordBatch, rows: &mut Vec<AggregateRow>) -> Result<()> {
    let configs = required_column::<StringArray>(batch, "config")?;
    let peak_counts = required_column::<UInt64Array>(batch, "peak_count")?;
    let auroc = required_column::<Float64Array>(batch, "auroc")?;
    let auprc = required_column::<Float64Array>(batch, "auprc")?;
    for row in 0..batch.num_rows() {
        rows.push(AggregateRow {
            config: configs.value(row).to_string(),
            peak_count: peak_counts.value(row),
            auroc: cell_or_none(auroc, row),
            auprc: cell_or_none(auprc, row),
        });
    }
    Ok(())
}

/// Extract every per-class row from one record batch.
fn observe_per_class_batch(batch: &RecordBatch, rows: &mut Vec<PerClassRow>) -> Result<()> {
    let configs = required_column::<StringArray>(batch, "config")?;
    let peak_counts = required_column::<UInt64Array>(batch, "peak_count")?;
    let pathways = required_column::<StringArray>(batch, "pathway")?;
    let auroc = required_column::<Float64Array>(batch, "auroc")?;
    let auprc = required_column::<Float64Array>(batch, "auprc")?;
    for row in 0..batch.num_rows() {
        rows.push(PerClassRow {
            config: configs.value(row).to_string(),
            peak_count: peak_counts.value(row),
            pathway: pathways.value(row).to_string(),
            auroc: cell_or_none(auroc, row),
            auprc: cell_or_none(auprc, row),
        });
    }
    Ok(())
}

/// Return `Some(value)` only when the cell is non-null and finite.
fn cell_or_none(column: &Float64Array, row: usize) -> Option<f64> {
    if column.is_null(row) {
        return None;
    }
    let value = column.value(row);
    if value.is_finite() { Some(value) } else { None }
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

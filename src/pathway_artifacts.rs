//! Derived artifacts for fixed-representative pathway predictions.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
    sync::Arc,
};

use anyhow::{Context, Result, bail};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, RecordBatch, StringArray, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use ndarray::{Array1, Array3, ArrayView2, Axis};
use ndarray_npy::NpzWriter;
use parquet::{arrow::ArrowWriter, arrow::arrow_reader::ParquetRecordBatchReaderBuilder};
use plotters::{
    coord::Shift,
    prelude::{
        BLACK, BitMapBackend, ChartBuilder, DrawingArea, DrawingBackend, IntoDrawingArea,
        LineSeries, PathElement, Rectangle, SVGBackend, SeriesLabelPosition, WHITE,
    },
    style::{Color, IntoFont, Palette, Palette99, RGBColor},
};

use crate::{
    model::PEAK_COUNT_GRID_SIZE,
    pathway::pathway_labels,
    progress::{ProgressTask, ScanProgress},
    visualize::{ensure_heatmap_font, plotters_error, sanitize_path_component},
};

/// Label used for missing predictions in categorical distribution summaries.
const UNASSIGNED_PREDICTION: &str = "__unassigned__";
/// Pseudo-pathway label used for support-weighted average metric rows.
const WEIGHTED_AVERAGE_PATHWAY: &str = "weighted_average";
/// Number of rendered categorical prediction-distribution heatmaps per config.
const PREDICTION_HEATMAP_METRICS: usize = 3;
/// Width of pathway prediction heatmap images in pixels.
const HEATMAP_WIDTH: u32 = 1_000;
/// Height of pathway prediction heatmap images in pixels.
const HEATMAP_HEIGHT: u32 = 900;
/// Width of the heatmap matrix panel before the colorbar.
const HEATMAP_CHART_WIDTH: u32 = 860;
/// Number of colorbar rectangles in heatmap legends.
const COLORBAR_STEPS: usize = 256;
/// Width of pathway metric line plots in pixels.
const LINE_PLOT_WIDTH: u32 = 1_200;
/// Height of pathway metric line plots in pixels.
const LINE_PLOT_HEIGHT: u32 = 760;
/// Color used for non-finite heatmap cells.
const NON_FINITE_COLOR: RGBColor = RGBColor(180, 180, 180);

/// Read `pathway_predictions.parquet` and write derived tables and plots.
pub fn write_pathway_prediction_artifacts(
    output_dir: &Path,
    progress: &ScanProgress,
) -> Result<()> {
    let predictions_path = output_dir.join("pathway_predictions.parquet");
    if !predictions_path.exists() {
        return Ok(());
    }

    let read_progress = progress.spinner("reading pathway predictions");
    let aggregation = read_prediction_aggregation(&predictions_path)?;
    read_progress.finish();
    if aggregation.is_empty() {
        return Ok(());
    }

    let build_progress = progress.spinner("building pathway prediction artifacts");
    let artifacts = aggregation.into_artifacts()?;
    build_progress.finish();

    let write_progress = progress.spinner("writing pathway prediction metric summaries");
    write_metric_rows(output_dir, &artifacts.metric_rows)?;
    write_progress.finish();

    let grid_progress = progress.spinner("writing pathway prediction distribution grids");
    write_distribution_grid_rows(output_dir, &artifacts.distribution_rows)?;
    write_distribution_grid_npz(output_dir, &artifacts.arrays)?;
    write_distribution_grid_configs(output_dir, &artifacts.configs)?;
    grid_progress.finish();

    write_prediction_distribution_heatmaps(
        output_dir,
        &artifacts.configs,
        &artifacts.arrays,
        progress,
    )?;
    write_metric_line_plots(
        output_dir,
        &artifacts.configs,
        &artifacts.metric_rows,
        progress,
    )
}

/// Read pathway prediction rows into compact categorical and confusion counts.
fn read_prediction_aggregation(path: &Path) -> Result<PredictionAggregation> {
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading metadata from {}", path.display()))?
        .build()
        .with_context(|| format!("building reader for {}", path.display()))?;

    let mut aggregation = PredictionAggregation::default();
    for batch in reader {
        aggregation.observe_batch(&batch?)?;
    }
    Ok(aggregation)
}

/// Aggregated prediction data keyed by similarity configuration.
#[derive(Default)]
struct PredictionAggregation {
    /// Per-config data in first-seen order.
    configs: Vec<ConfigPredictionData>,
    /// Map from dataset/config key to index in `configs`.
    indices: BTreeMap<String, usize>,
}

impl PredictionAggregation {
    /// Return whether no prediction rows were observed.
    fn is_empty(&self) -> bool {
        self.configs.is_empty()
    }

    /// Add all rows from one Arrow record batch.
    fn observe_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        let datasets = required_column::<StringArray>(batch, "dataset")?;
        let configs = required_column::<StringArray>(batch, "config")?;
        let peak_counts = required_column::<UInt64Array>(batch, "peak_count")?;
        let expected = required_column::<StringArray>(batch, "query_npc_pathway")?;
        let predicted = required_column::<StringArray>(batch, "predicted_npc_pathway")?;

        for row in 0..batch.num_rows() {
            let peak_count =
                usize::try_from(peak_counts.value(row)).context("peak_count does not fit usize")?;
            let config_index = self.config_index(datasets.value(row), configs.value(row));
            self.configs[config_index].observe(
                peak_count,
                optional_string(expected, row),
                optional_string(predicted, row),
            )?;
        }
        Ok(())
    }

    /// Return the stable config index for one dataset/config pair.
    fn config_index(&mut self, dataset: &str, config: &str) -> usize {
        let key = format!("{dataset}\u{0}{config}");
        if let Some(index) = self.indices.get(&key) {
            return *index;
        }

        let index = self.configs.len();
        self.indices.insert(key, index);
        self.configs
            .push(ConfigPredictionData::new(dataset, config));
        index
    }

    /// Build all tabular rows and dense matrices from the aggregation.
    fn into_artifacts(self) -> Result<PathwayPredictionArtifacts> {
        let mut metric_rows = Vec::new();
        let mut distribution_rows = Vec::new();
        let mut config_rows = Vec::with_capacity(self.configs.len());
        let shape = (
            self.configs.len(),
            PEAK_COUNT_GRID_SIZE,
            PEAK_COUNT_GRID_SIZE,
        );
        let mut total_variation = Array3::<f64>::from_elem(shape, f64::NAN);
        let mut jensen_shannon_distance = Array3::<f64>::from_elem(shape, f64::NAN);
        let mut hellinger_distance = Array3::<f64>::from_elem(shape, f64::NAN);

        for (config_index, config) in self.configs.iter().enumerate() {
            config_rows.push(PathwayConfigRow {
                config_index,
                dataset: config.dataset.clone(),
                config: config.config.clone(),
            });
            metric_rows.extend(config.metric_rows());
            for peak_a in 1..=PEAK_COUNT_GRID_SIZE {
                for peak_b in 1..=PEAK_COUNT_GRID_SIZE {
                    let index_a = peak_a - 1;
                    let index_b = peak_b - 1;
                    let comparison = compare_prediction_counts(
                        &config.prediction_counts[index_a],
                        &config.prediction_counts[index_b],
                    );
                    total_variation[[config_index, index_a, index_b]] = comparison.total_variation;
                    jensen_shannon_distance[[config_index, index_a, index_b]] =
                        comparison.jensen_shannon_distance;
                    hellinger_distance[[config_index, index_a, index_b]] =
                        comparison.hellinger_distance;
                    distribution_rows.push(PathwayDistributionRow {
                        dataset: config.dataset.clone(),
                        config: config.config.clone(),
                        peak_count_a: peak_a,
                        peak_count_b: peak_b,
                        n_predictions_a: comparison.n_predictions_a,
                        n_predictions_b: comparison.n_predictions_b,
                        total_variation: comparison.total_variation,
                        jensen_shannon_distance: comparison.jensen_shannon_distance,
                        hellinger_distance: comparison.hellinger_distance,
                    });
                }
            }
        }

        Ok(PathwayPredictionArtifacts {
            configs: config_rows,
            metric_rows,
            distribution_rows,
            arrays: PathwayDistributionArrays {
                peak_counts: peak_counts_array()?,
                total_variation,
                jensen_shannon_distance,
                hellinger_distance,
            },
        })
    }
}

/// Per-config compact counts for prediction distributions and pathway metrics.
struct ConfigPredictionData {
    /// Dataset label.
    dataset: String,
    /// Similarity configuration label.
    config: String,
    /// Prediction label counts by retained peak count.
    prediction_counts: Vec<BTreeMap<String, usize>>,
    /// Number of labeled queries by retained peak count.
    labeled_totals: Vec<usize>,
    /// Expected pathway support by retained peak count.
    actual_supports: Vec<BTreeMap<String, usize>>,
    /// Predicted pathway counts over labeled queries by retained peak count.
    predicted_counts: Vec<BTreeMap<String, usize>>,
    /// True positives by predicted pathway and retained peak count.
    true_positives: Vec<BTreeMap<String, usize>>,
    /// All expected pathway labels observed for the config.
    pathways: BTreeSet<String>,
}

impl ConfigPredictionData {
    /// Create empty counters for one dataset/config pair.
    fn new(dataset: &str, config: &str) -> Self {
        Self {
            dataset: dataset.to_string(),
            config: config.to_string(),
            prediction_counts: repeated_maps(),
            labeled_totals: vec![0; PEAK_COUNT_GRID_SIZE],
            actual_supports: repeated_maps(),
            predicted_counts: repeated_maps(),
            true_positives: repeated_maps(),
            pathways: BTreeSet::new(),
        }
    }

    /// Observe one prediction row.
    fn observe(
        &mut self,
        peak_count: usize,
        raw_expected: Option<&str>,
        raw_predicted: Option<&str>,
    ) -> Result<()> {
        let peak_index = peak_count
            .checked_sub(1)
            .with_context(|| format!("invalid peak count {peak_count}"))?;
        if peak_index >= PEAK_COUNT_GRID_SIZE {
            bail!("peak count {peak_count} exceeds {PEAK_COUNT_GRID_SIZE}");
        }

        let predicted = raw_predicted
            .map(str::trim)
            .filter(|label| !label.is_empty())
            .unwrap_or(UNASSIGNED_PREDICTION)
            .to_string();
        increment(&mut self.prediction_counts[peak_index], &predicted);

        let expected_labels = pathway_labels(raw_expected);
        if expected_labels.is_empty() {
            return Ok(());
        }
        self.labeled_totals[peak_index] += 1;
        if predicted != UNASSIGNED_PREDICTION {
            increment(&mut self.predicted_counts[peak_index], &predicted);
        }

        let predicted_is_true_positive = expected_labels.iter().any(|label| label == &predicted);
        for label in expected_labels {
            self.pathways.insert(label.clone());
            increment(&mut self.actual_supports[peak_index], &label);
        }
        if predicted != UNASSIGNED_PREDICTION && predicted_is_true_positive {
            increment(&mut self.true_positives[peak_index], &predicted);
        }
        Ok(())
    }

    /// Build per-pathway and support-weighted metric rows for this config.
    #[allow(clippy::similar_names)]
    fn metric_rows(&self) -> Vec<PathwayMetricRow> {
        let mut rows = Vec::new();
        for peak_index in 0..PEAK_COUNT_GRID_SIZE {
            let peak_count = peak_index + 1;
            let n_labeled = self.labeled_totals[peak_index];
            let mut weighted_accuracy_sum = 0.0;
            let mut weighted_mcc_sum = 0.0;
            let mut total_support = 0_usize;
            let mut total_tp = 0_usize;
            let mut total_fp = 0_usize;
            let mut total_tn = 0_usize;
            let mut total_fn = 0_usize;

            for pathway in &self.pathways {
                let support = count(&self.actual_supports[peak_index], pathway);
                if support == 0 {
                    continue;
                }
                let tp = count(&self.true_positives[peak_index], pathway);
                let predicted = count(&self.predicted_counts[peak_index], pathway);
                let fp = predicted.saturating_sub(tp);
                let fn_count = support.saturating_sub(tp);
                let tn = n_labeled.saturating_sub(tp + fp + fn_count);
                let accuracy = accuracy(tp, fp, tn, fn_count);
                let mcc = matthews_correlation(tp, fp, tn, fn_count);

                weighted_accuracy_sum = accuracy.mul_add(support as f64, weighted_accuracy_sum);
                weighted_mcc_sum = mcc.mul_add(support as f64, weighted_mcc_sum);
                total_support += support;
                total_tp += tp;
                total_fp += fp;
                total_tn += tn;
                total_fn += fn_count;

                rows.push(PathwayMetricRow {
                    dataset: self.dataset.clone(),
                    config: self.config.clone(),
                    peak_count,
                    pathway: pathway.clone(),
                    is_weighted_average: false,
                    n_labeled,
                    support,
                    true_positive: tp,
                    false_positive: fp,
                    true_negative: tn,
                    false_negative: fn_count,
                    accuracy,
                    mcc,
                });
            }

            if total_support > 0 {
                rows.push(PathwayMetricRow {
                    dataset: self.dataset.clone(),
                    config: self.config.clone(),
                    peak_count,
                    pathway: WEIGHTED_AVERAGE_PATHWAY.to_string(),
                    is_weighted_average: true,
                    n_labeled,
                    support: total_support,
                    true_positive: total_tp,
                    false_positive: total_fp,
                    true_negative: total_tn,
                    false_negative: total_fn,
                    accuracy: weighted_accuracy_sum / total_support as f64,
                    mcc: weighted_mcc_sum / total_support as f64,
                });
            }
        }
        rows
    }
}

/// Complete derived artifact data before serialization.
struct PathwayPredictionArtifacts {
    /// Config metadata rows.
    configs: Vec<PathwayConfigRow>,
    /// Per-pathway and weighted metric rows.
    metric_rows: Vec<PathwayMetricRow>,
    /// Full categorical prediction-distribution comparison rows.
    distribution_rows: Vec<PathwayDistributionRow>,
    /// Dense categorical prediction-distribution distance arrays.
    arrays: PathwayDistributionArrays,
}

/// Config-axis metadata for pathway prediction distribution arrays.
struct PathwayConfigRow {
    /// Zero-based config index.
    config_index: usize,
    /// Dataset label.
    dataset: String,
    /// Similarity configuration label.
    config: String,
}

/// Per-pathway metric row for one config and retained peak count.
struct PathwayMetricRow {
    /// Dataset label.
    dataset: String,
    /// Similarity configuration label.
    config: String,
    /// Retained peak count.
    peak_count: usize,
    /// Pathway label, or `weighted_average`.
    pathway: String,
    /// Whether this row is the support-weighted average over pathways.
    is_weighted_average: bool,
    /// Number of labeled query rows used at this peak count.
    n_labeled: usize,
    /// Positive class support for this pathway.
    support: usize,
    /// One-vs-rest true positives.
    true_positive: usize,
    /// One-vs-rest false positives.
    false_positive: usize,
    /// One-vs-rest true negatives.
    true_negative: usize,
    /// One-vs-rest false negatives.
    false_negative: usize,
    /// One-vs-rest accuracy.
    accuracy: f64,
    /// One-vs-rest Matthews correlation coefficient.
    mcc: f64,
}

/// Full-grid comparison of categorical prediction distributions.
struct PathwayDistributionRow {
    /// Dataset label.
    dataset: String,
    /// Similarity configuration label.
    config: String,
    /// First retained peak count.
    peak_count_a: usize,
    /// Second retained peak count.
    peak_count_b: usize,
    /// Number of predictions in the first distribution.
    n_predictions_a: usize,
    /// Number of predictions in the second distribution.
    n_predictions_b: usize,
    /// Total variation distance.
    total_variation: f64,
    /// Jensen-Shannon distance using base-2 logarithms.
    jensen_shannon_distance: f64,
    /// Hellinger distance.
    hellinger_distance: f64,
}

/// Dense categorical prediction-distribution distance arrays.
struct PathwayDistributionArrays {
    /// One-based retained peak-count axis.
    peak_counts: Array1<u64>,
    /// Total variation distance matrix.
    total_variation: Array3<f64>,
    /// Jensen-Shannon distance matrix.
    jensen_shannon_distance: Array3<f64>,
    /// Hellinger distance matrix.
    hellinger_distance: Array3<f64>,
}

/// Distance values for comparing two categorical prediction distributions.
struct PredictionDistributionComparison {
    /// Number of predictions in the first distribution.
    n_predictions_a: usize,
    /// Number of predictions in the second distribution.
    n_predictions_b: usize,
    /// Total variation distance.
    total_variation: f64,
    /// Jensen-Shannon distance using base-2 logarithms.
    jensen_shannon_distance: f64,
    /// Hellinger distance.
    hellinger_distance: f64,
}

/// Return a vector of empty maps, one per retained peak count.
fn repeated_maps() -> Vec<BTreeMap<String, usize>> {
    (0..PEAK_COUNT_GRID_SIZE).map(|_| BTreeMap::new()).collect()
}

/// Increment one string-keyed counter.
fn increment(counts: &mut BTreeMap<String, usize>, key: &str) {
    *counts.entry(key.to_string()).or_default() += 1;
}

/// Return the value for one counter key.
fn count(counts: &BTreeMap<String, usize>, key: &str) -> usize {
    counts.get(key).copied().unwrap_or(0)
}

/// Compute one-vs-rest accuracy.
fn accuracy(tp: usize, fp: usize, tn: usize, fn_count: usize) -> f64 {
    let total = tp + fp + tn + fn_count;
    if total == 0 {
        return f64::NAN;
    }
    (tp + tn) as f64 / total as f64
}

/// Compute the Matthews correlation coefficient.
fn matthews_correlation(tp: usize, fp: usize, tn: usize, fn_count: usize) -> f64 {
    let numerator = (tp * tn) as f64 - (fp * fn_count) as f64;
    let denominator =
        ((tp + fp) as f64 * (tp + fn_count) as f64 * (tn + fp) as f64 * (tn + fn_count) as f64)
            .sqrt();
    if denominator <= f64::EPSILON {
        0.0
    } else {
        numerator / denominator
    }
}

/// Compare two categorical prediction-count maps.
fn compare_prediction_counts(
    counts_a: &BTreeMap<String, usize>,
    counts_b: &BTreeMap<String, usize>,
) -> PredictionDistributionComparison {
    let n_predictions_a = counts_a.values().sum::<usize>();
    let n_predictions_b = counts_b.values().sum::<usize>();
    if n_predictions_a == 0 || n_predictions_b == 0 {
        return PredictionDistributionComparison {
            n_predictions_a,
            n_predictions_b,
            total_variation: f64::NAN,
            jensen_shannon_distance: f64::NAN,
            hellinger_distance: f64::NAN,
        };
    }

    let labels = counts_a
        .keys()
        .chain(counts_b.keys())
        .collect::<BTreeSet<_>>();
    let mut total_variation = 0.0;
    let mut jensen_shannon_divergence = 0.0;
    let mut hellinger_sum = 0.0;
    for label in labels {
        let probability_a = count(counts_a, label) as f64 / n_predictions_a as f64;
        let probability_b = count(counts_b, label) as f64 / n_predictions_b as f64;
        total_variation += (probability_a - probability_b).abs();
        let midpoint = 0.5 * (probability_a + probability_b);
        if probability_a > 0.0 {
            jensen_shannon_divergence = (0.5 * probability_a)
                .mul_add((probability_a / midpoint).log2(), jensen_shannon_divergence);
        }
        if probability_b > 0.0 {
            jensen_shannon_divergence = (0.5 * probability_b)
                .mul_add((probability_b / midpoint).log2(), jensen_shannon_divergence);
        }
        hellinger_sum += (probability_a.sqrt() - probability_b.sqrt()).powi(2);
    }

    PredictionDistributionComparison {
        n_predictions_a,
        n_predictions_b,
        total_variation: 0.5 * total_variation,
        jensen_shannon_distance: jensen_shannon_divergence.max(0.0).sqrt(),
        hellinger_distance: (0.5 * hellinger_sum).sqrt(),
    }
}

/// Write the per-pathway metric summary table.
fn write_metric_rows(output_dir: &Path, rows: &[PathwayMetricRow]) -> Result<()> {
    write_parquet(
        &output_dir.join("pathway_prediction_metrics.parquet"),
        pathway_metric_schema(),
        &pathway_metric_batch(rows)?,
    )
}

/// Write the full categorical prediction-distribution comparison table.
fn write_distribution_grid_rows(output_dir: &Path, rows: &[PathwayDistributionRow]) -> Result<()> {
    write_parquet(
        &output_dir.join("pathway_prediction_distribution_grid.parquet"),
        pathway_distribution_schema(),
        &pathway_distribution_batch(rows)?,
    )
}

/// Write dense categorical prediction-distribution arrays.
fn write_distribution_grid_npz(
    output_dir: &Path,
    arrays: &PathwayDistributionArrays,
) -> Result<()> {
    let path = output_dir.join("pathway_prediction_distribution_grid.npz");
    let file = fs::File::create(&path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = NpzWriter::new(file);
    writer.add_array("peak_counts", &arrays.peak_counts)?;
    writer.add_array("total_variation", &arrays.total_variation)?;
    writer.add_array("jensen_shannon_distance", &arrays.jensen_shannon_distance)?;
    writer.add_array("hellinger_distance", &arrays.hellinger_distance)?;
    writer.finish()?;
    Ok(())
}

/// Write config-axis metadata for pathway prediction distribution arrays.
fn write_distribution_grid_configs(output_dir: &Path, rows: &[PathwayConfigRow]) -> Result<()> {
    write_parquet(
        &output_dir.join("pathway_prediction_distribution_grid_configs.parquet"),
        pathway_config_schema(),
        &pathway_config_batch(rows)?,
    )
}

/// Write one Parquet file containing a single record batch.
fn write_parquet(path: &Path, schema: SchemaRef, batch: &RecordBatch) -> Result<()> {
    let file = fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, None)
        .with_context(|| format!("opening Parquet writer for {}", path.display()))?;
    if batch.num_rows() > 0 {
        writer.write(batch)?;
    }
    writer.close()?;
    Ok(())
}

/// Build one record batch for pathway metric rows.
fn pathway_metric_batch(rows: &[PathwayMetricRow]) -> Result<RecordBatch> {
    let columns = vec![
        strings(rows.iter().map(|row| row.dataset.as_str()).collect()),
        strings(rows.iter().map(|row| row.config.as_str()).collect()),
        usizes(rows.iter().map(|row| row.peak_count).collect())?,
        strings(rows.iter().map(|row| row.pathway.as_str()).collect()),
        booleans(rows.iter().map(|row| row.is_weighted_average).collect()),
        usizes(rows.iter().map(|row| row.n_labeled).collect())?,
        usizes(rows.iter().map(|row| row.support).collect())?,
        usizes(rows.iter().map(|row| row.true_positive).collect())?,
        usizes(rows.iter().map(|row| row.false_positive).collect())?,
        usizes(rows.iter().map(|row| row.true_negative).collect())?,
        usizes(rows.iter().map(|row| row.false_negative).collect())?,
        floats(rows.iter().map(|row| row.accuracy).collect()),
        floats(rows.iter().map(|row| row.mcc).collect()),
    ];
    RecordBatch::try_new(pathway_metric_schema(), columns)
        .context("building pathway prediction metric batch")
}

/// Build one record batch for pathway distribution rows.
fn pathway_distribution_batch(rows: &[PathwayDistributionRow]) -> Result<RecordBatch> {
    let columns = vec![
        strings(rows.iter().map(|row| row.dataset.as_str()).collect()),
        strings(rows.iter().map(|row| row.config.as_str()).collect()),
        usizes(rows.iter().map(|row| row.peak_count_a).collect())?,
        usizes(rows.iter().map(|row| row.peak_count_b).collect())?,
        usizes(rows.iter().map(|row| row.n_predictions_a).collect())?,
        usizes(rows.iter().map(|row| row.n_predictions_b).collect())?,
        floats(rows.iter().map(|row| row.total_variation).collect()),
        floats(rows.iter().map(|row| row.jensen_shannon_distance).collect()),
        floats(rows.iter().map(|row| row.hellinger_distance).collect()),
    ];
    RecordBatch::try_new(pathway_distribution_schema(), columns)
        .context("building pathway prediction distribution batch")
}

/// Build one record batch for pathway config-axis rows.
fn pathway_config_batch(rows: &[PathwayConfigRow]) -> Result<RecordBatch> {
    let columns = vec![
        usizes(rows.iter().map(|row| row.config_index).collect())?,
        strings(rows.iter().map(|row| row.dataset.as_str()).collect()),
        strings(rows.iter().map(|row| row.config.as_str()).collect()),
    ];
    RecordBatch::try_new(pathway_config_schema(), columns)
        .context("building pathway prediction distribution config batch")
}

/// Return the pathway metric summary schema.
fn pathway_metric_schema() -> SchemaRef {
    schema(vec![
        utf8("dataset", false),
        utf8("config", false),
        u64_field("peak_count", false),
        utf8("pathway", false),
        bool_field("is_weighted_average", false),
        u64_field("n_labeled", false),
        u64_field("support", false),
        u64_field("true_positive", false),
        u64_field("false_positive", false),
        u64_field("true_negative", false),
        u64_field("false_negative", false),
        f64_field("accuracy", false),
        f64_field("mcc", false),
    ])
}

/// Return the pathway prediction distribution grid schema.
fn pathway_distribution_schema() -> SchemaRef {
    schema(vec![
        utf8("dataset", false),
        utf8("config", false),
        u64_field("peak_count_a", false),
        u64_field("peak_count_b", false),
        u64_field("n_predictions_a", false),
        u64_field("n_predictions_b", false),
        f64_field("total_variation", false),
        f64_field("jensen_shannon_distance", false),
        f64_field("hellinger_distance", false),
    ])
}

/// Return the pathway prediction distribution config schema.
fn pathway_config_schema() -> SchemaRef {
    schema(vec![
        u64_field("config_index", false),
        utf8("dataset", false),
        utf8("config", false),
    ])
}

/// Build an Arrow schema from fields.
fn schema(fields: Vec<Field>) -> SchemaRef {
    Arc::new(Schema::new(fields))
}

/// Build a UTF-8 field.
fn utf8(name: &str, nullable: bool) -> Field {
    Field::new(name, DataType::Utf8, nullable)
}

/// Build a `u64` field.
fn u64_field(name: &str, nullable: bool) -> Field {
    Field::new(name, DataType::UInt64, nullable)
}

/// Build an `f64` field.
fn f64_field(name: &str, nullable: bool) -> Field {
    Field::new(name, DataType::Float64, nullable)
}

/// Build a boolean field.
fn bool_field(name: &str, nullable: bool) -> Field {
    Field::new(name, DataType::Boolean, nullable)
}

/// Build a required UTF-8 array.
fn strings(values: Vec<&str>) -> ArrayRef {
    Arc::new(StringArray::from(values))
}

/// Build a required boolean array.
fn booleans(values: Vec<bool>) -> ArrayRef {
    Arc::new(BooleanArray::from(values))
}

/// Build a required `f64` array.
fn floats(values: Vec<f64>) -> ArrayRef {
    Arc::new(Float64Array::from(values))
}

/// Build a required `u64` array from `usize` values.
fn usizes(values: Vec<usize>) -> Result<ArrayRef> {
    Ok(Arc::new(UInt64Array::from(
        values
            .into_iter()
            .map(usize_to_u64)
            .collect::<Result<Vec<_>>>()?,
    )))
}

/// Convert a `usize` into `u64`.
fn usize_to_u64(value: usize) -> Result<u64> {
    u64::try_from(value).context("usize value does not fit u64")
}

/// Return the `1..=128` peak-count axis as a `NumPy` array.
fn peak_counts_array() -> Result<Array1<u64>> {
    let values = (1..=PEAK_COUNT_GRID_SIZE)
        .map(usize_to_u64)
        .collect::<Result<Vec<_>>>()?;
    Ok(Array1::from(values))
}

/// Return a typed required Arrow column from a record batch.
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

/// Return a nullable string value from a `StringArray`.
fn optional_string(array: &StringArray, row: usize) -> Option<&str> {
    (!array.is_null(row)).then(|| array.value(row))
}

/// Render categorical prediction-distribution heatmaps.
fn write_prediction_distribution_heatmaps(
    output_dir: &Path,
    configs: &[PathwayConfigRow],
    arrays: &PathwayDistributionArrays,
    progress: &ScanProgress,
) -> Result<()> {
    ensure_heatmap_font()?;
    let output_dir = output_dir.join("pathway_prediction_heatmaps");
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("creating {}", output_dir.display()))?;
    let total = configs
        .len()
        .checked_mul(PREDICTION_HEATMAP_METRICS)
        .and_then(|count| count.checked_mul(2))
        .unwrap_or(usize::MAX);
    let task = progress.bar(
        u64::try_from(total).unwrap_or(u64::MAX),
        "rendering pathway prediction heatmaps",
    );

    for (config_index, config) in configs.iter().enumerate() {
        let config_dir = output_dir.join(sanitize_path_component(&config.config));
        fs::create_dir_all(&config_dir)
            .with_context(|| format!("creating {}", config_dir.display()))?;
        for metric in prediction_heatmap_metrics(arrays, config_index) {
            task.set_message(format!(
                "rendering pathway prediction {} {}",
                config.config, metric.name
            ));
            write_prediction_heatmap_pair(&config_dir, &config.config, &metric, &task)?;
        }
    }
    task.finish();
    Ok(())
}

/// Return heatmap metric views for one config.
fn prediction_heatmap_metrics(
    arrays: &PathwayDistributionArrays,
    config_index: usize,
) -> [PredictionHeatmapMetric<'_>; 3] {
    [
        PredictionHeatmapMetric {
            name: "total_variation",
            title: "Prediction total variation",
            values: arrays.total_variation.index_axis(Axis(0), config_index),
        },
        PredictionHeatmapMetric {
            name: "jensen_shannon_distance",
            title: "Prediction Jensen-Shannon distance",
            values: arrays
                .jensen_shannon_distance
                .index_axis(Axis(0), config_index),
        },
        PredictionHeatmapMetric {
            name: "hellinger_distance",
            title: "Prediction Hellinger distance",
            values: arrays.hellinger_distance.index_axis(Axis(0), config_index),
        },
    ]
}

/// One categorical prediction-distribution heatmap metric.
struct PredictionHeatmapMetric<'a> {
    /// Stable artifact file stem.
    name: &'static str,
    /// Human-readable title.
    title: &'static str,
    /// Matrix values for one config.
    values: ArrayView2<'a, f64>,
}

/// Write SVG and PNG versions of one categorical prediction heatmap.
fn write_prediction_heatmap_pair(
    output_dir: &Path,
    config: &str,
    metric: &PredictionHeatmapMetric<'_>,
    progress: &ProgressTask,
) -> Result<()> {
    let stem = output_dir.join(metric.name);
    write_prediction_heatmap_svg(&stem.with_extension("svg"), config, metric)?;
    progress.inc(1);
    write_prediction_heatmap_png(&stem.with_extension("png"), config, metric)?;
    progress.inc(1);
    Ok(())
}

/// Write one categorical prediction heatmap as SVG.
fn write_prediction_heatmap_svg(
    path: &Path,
    config: &str,
    metric: &PredictionHeatmapMetric<'_>,
) -> Result<()> {
    let root = SVGBackend::new(path, (HEATMAP_WIDTH, HEATMAP_HEIGHT)).into_drawing_area();
    draw_prediction_heatmap(&root, config, metric)
        .with_context(|| format!("writing SVG heatmap {}", path.display()))
}

/// Write one categorical prediction heatmap as PNG.
fn write_prediction_heatmap_png(
    path: &Path,
    config: &str,
    metric: &PredictionHeatmapMetric<'_>,
) -> Result<()> {
    let root = BitMapBackend::new(path, (HEATMAP_WIDTH, HEATMAP_HEIGHT)).into_drawing_area();
    draw_prediction_heatmap(&root, config, metric)
        .with_context(|| format!("writing PNG heatmap {}", path.display()))
}

/// Draw one categorical prediction heatmap.
fn draw_prediction_heatmap<Backend>(
    root: &DrawingArea<Backend, Shift>,
    config: &str,
    metric: &PredictionHeatmapMetric<'_>,
) -> Result<()>
where
    Backend: DrawingBackend,
    Backend::ErrorType: std::fmt::Debug,
{
    root.fill(&WHITE).map_err(plotters_error)?;
    let (chart_area, colorbar_area) = root.split_horizontally(HEATMAP_CHART_WIDTH);
    draw_prediction_matrix(&chart_area, config, metric)?;
    draw_prediction_colorbar(&colorbar_area)?;
    root.present().map_err(plotters_error)
}

/// Draw the categorical prediction heatmap matrix.
fn draw_prediction_matrix<Backend>(
    area: &DrawingArea<Backend, Shift>,
    config: &str,
    metric: &PredictionHeatmapMetric<'_>,
) -> Result<()>
where
    Backend: DrawingBackend,
    Backend::ErrorType: std::fmt::Debug,
{
    let x_end = usize_to_i32(metric.values.ncols() + 1)?;
    let y_end = usize_to_i32(metric.values.nrows() + 1)?;
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
        .draw_series(prediction_matrix_cells(metric)?)
        .map_err(plotters_error)?;
    Ok(())
}

/// Draw the categorical prediction heatmap colorbar.
fn draw_prediction_colorbar<Backend>(area: &DrawingArea<Backend, Shift>) -> Result<()>
where
    Backend: DrawingBackend,
    Backend::ErrorType: std::fmt::Debug,
{
    let mut chart = ChartBuilder::on(area)
        .caption("distance", ("sans-serif", 16))
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
        .y_labels(6)
        .label_style(("sans-serif", 15))
        .y_label_formatter(&|position| format!("{position:.2}"))
        .draw()
        .map_err(plotters_error)?;
    chart
        .draw_series((0..COLORBAR_STEPS).map(|step| {
            let lower = step as f64 / COLORBAR_STEPS as f64;
            let upper = (step + 1) as f64 / COLORBAR_STEPS as f64;
            let sample = f64::midpoint(lower, upper);
            Rectangle::new(
                [(0.0, lower), (1.0, upper)],
                distance_color(sample).filled(),
            )
        }))
        .map_err(plotters_error)?;
    Ok(())
}

/// Return matrix cells for a categorical prediction heatmap.
fn prediction_matrix_cells(
    metric: &PredictionHeatmapMetric<'_>,
) -> Result<Vec<Rectangle<(i32, i32)>>> {
    let mut cells = Vec::with_capacity(metric.values.len());
    for row in 0..metric.values.nrows() {
        for column in 0..metric.values.ncols() {
            let x0 = usize_to_i32(column + 1)?;
            let y0 = usize_to_i32(row + 1)?;
            let value = metric.values[[row, column]];
            let color = if value.is_finite() {
                distance_color(value)
            } else {
                NON_FINITE_COLOR
            };
            cells.push(Rectangle::new([(x0, y0), (x0 + 1, y0 + 1)], color.filled()));
        }
    }
    Ok(cells)
}

/// Return a Viridis color for a normalized distance.
fn distance_color(value: f64) -> RGBColor {
    let color = colorous::VIRIDIS.eval_continuous(value.clamp(0.0, 1.0));
    RGBColor(color.r, color.g, color.b)
}

/// Render pathway accuracy and MCC line plots.
fn write_metric_line_plots(
    output_dir: &Path,
    configs: &[PathwayConfigRow],
    rows: &[PathwayMetricRow],
    progress: &ScanProgress,
) -> Result<()> {
    ensure_heatmap_font()?;
    let output_dir = output_dir.join("pathway_prediction_plots");
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("creating {}", output_dir.display()))?;
    let total = configs.len().saturating_mul(4);
    let task = progress.bar(
        u64::try_from(total).unwrap_or(u64::MAX),
        "rendering pathway prediction metric plots",
    );

    for config in configs {
        let config_dir = output_dir.join(sanitize_path_component(&config.config));
        fs::create_dir_all(&config_dir)
            .with_context(|| format!("creating {}", config_dir.display()))?;
        let config_rows = rows
            .iter()
            .filter(|row| row.config == config.config && row.dataset == config.dataset)
            .collect::<Vec<_>>();
        for metric in [
            LineMetric::Accuracy,
            LineMetric::MatthewsCorrelationCoefficient,
        ] {
            task.set_message(format!(
                "rendering {} {}",
                config.config,
                metric.file_stem()
            ));
            write_line_plot_pair(&config_dir, &config.config, metric, &config_rows, &task)?;
        }
    }
    task.finish();
    Ok(())
}

/// Metric rendered as a pathway line plot.
#[derive(Clone, Copy)]
enum LineMetric {
    /// One-vs-rest accuracy.
    Accuracy,
    /// One-vs-rest Matthews correlation coefficient.
    MatthewsCorrelationCoefficient,
}

impl LineMetric {
    /// Return the artifact file stem.
    const fn file_stem(self) -> &'static str {
        match self {
            Self::Accuracy => "accuracy",
            Self::MatthewsCorrelationCoefficient => "mcc",
        }
    }

    /// Return the plot title fragment.
    const fn title(self) -> &'static str {
        match self {
            Self::Accuracy => "Pathway accuracy",
            Self::MatthewsCorrelationCoefficient => "Pathway MCC",
        }
    }

    /// Return the y-axis label.
    const fn y_label(self) -> &'static str {
        match self {
            Self::Accuracy => "Accuracy",
            Self::MatthewsCorrelationCoefficient => "MCC",
        }
    }

    /// Return the fixed y-axis range.
    const fn y_range(self) -> (f64, f64) {
        match self {
            Self::Accuracy => (0.0, 1.0),
            Self::MatthewsCorrelationCoefficient => (-1.0, 1.0),
        }
    }

    /// Return this metric value from a row.
    const fn value(self, row: &PathwayMetricRow) -> f64 {
        match self {
            Self::Accuracy => row.accuracy,
            Self::MatthewsCorrelationCoefficient => row.mcc,
        }
    }
}

/// Write SVG and PNG versions of one line plot.
fn write_line_plot_pair(
    output_dir: &Path,
    config: &str,
    metric: LineMetric,
    rows: &[&PathwayMetricRow],
    progress: &ProgressTask,
) -> Result<()> {
    let stem = output_dir.join(metric.file_stem());
    write_line_plot_svg(&stem.with_extension("svg"), config, metric, rows)?;
    progress.inc(1);
    write_line_plot_png(&stem.with_extension("png"), config, metric, rows)?;
    progress.inc(1);
    Ok(())
}

/// Write one line plot as SVG.
fn write_line_plot_svg(
    path: &Path,
    config: &str,
    metric: LineMetric,
    rows: &[&PathwayMetricRow],
) -> Result<()> {
    let root = SVGBackend::new(path, (LINE_PLOT_WIDTH, LINE_PLOT_HEIGHT)).into_drawing_area();
    draw_line_plot(&root, config, metric, rows)
        .with_context(|| format!("writing SVG line plot {}", path.display()))
}

/// Write one line plot as PNG.
fn write_line_plot_png(
    path: &Path,
    config: &str,
    metric: LineMetric,
    rows: &[&PathwayMetricRow],
) -> Result<()> {
    let root = BitMapBackend::new(path, (LINE_PLOT_WIDTH, LINE_PLOT_HEIGHT)).into_drawing_area();
    draw_line_plot(&root, config, metric, rows)
        .with_context(|| format!("writing PNG line plot {}", path.display()))
}

/// Draw one pathway metric line plot.
fn draw_line_plot<Backend>(
    root: &DrawingArea<Backend, Shift>,
    config: &str,
    metric: LineMetric,
    rows: &[&PathwayMetricRow],
) -> Result<()>
where
    Backend: DrawingBackend,
    Backend::ErrorType: std::fmt::Debug,
{
    root.fill(&WHITE).map_err(plotters_error)?;
    let (y_min, y_max) = metric.y_range();
    let x_end = usize_to_i32(PEAK_COUNT_GRID_SIZE + 1)?;
    let mut chart = ChartBuilder::on(root)
        .caption(format!("{config} / {}", metric.title()), ("sans-serif", 24))
        .margin(24)
        .x_label_area_size(48)
        .y_label_area_size(64)
        .build_cartesian_2d(1_i32..x_end, y_min..y_max)
        .map_err(plotters_error)?;
    chart
        .configure_mesh()
        .x_desc("Peak count")
        .y_desc(metric.y_label())
        .x_labels(8)
        .y_labels(8)
        .axis_desc_style(("sans-serif", 20))
        .label_style(("sans-serif", 15))
        .draw()
        .map_err(plotters_error)?;

    let mut pathways = rows
        .iter()
        .filter(|row| !row.is_weighted_average)
        .map(|row| row.pathway.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    pathways.push(WEIGHTED_AVERAGE_PATHWAY);

    for (index, pathway) in pathways.iter().enumerate() {
        let points = rows
            .iter()
            .filter(|row| row.pathway == *pathway)
            .filter_map(|row| {
                let value = metric.value(row);
                value
                    .is_finite()
                    .then_some((usize_to_i32(row.peak_count).ok()?, value))
            })
            .collect::<Vec<_>>();
        if points.is_empty() {
            continue;
        }
        let style = if *pathway == WEIGHTED_AVERAGE_PATHWAY {
            BLACK.stroke_width(4)
        } else {
            Palette99::pick(index).stroke_width(2)
        };
        chart
            .draw_series(LineSeries::new(points, style))
            .map_err(plotters_error)?
            .label((*pathway).to_string())
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 28, y)], style));
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.82))
        .border_style(BLACK)
        .label_font(("sans-serif", 14).into_font())
        .draw()
        .map_err(plotters_error)?;
    root.present().map_err(plotters_error)
}

/// Convert a `usize` to `i32` for plotting coordinates.
fn usize_to_i32(value: usize) -> Result<i32> {
    i32::try_from(value).context("plot coordinate does not fit i32")
}

//! Parquet and NumPy artifact writers for completed scan outputs.

use std::{
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, bail};
use arrow_array::{ArrayRef, BooleanArray, Float64Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use ndarray::{Array1, Array3};
use ndarray_npy::NpzWriter;
use parquet::arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder};
use serde::{Deserialize, Serialize};

use crate::model::{
    DistributionComparison, DistributionHistogramBin, DistributionSummary, PEAK_COUNT_GRID_SIZE,
    PathwayPrediction, PathwayScore,
};
use crate::progress::ScanProgress;
use crate::visualize::write_heatmaps;

/// Directory under an output directory that stores pathway shard Parquet files.
const PATHWAY_SHARD_DIR: &str = "pathway_shards";
/// Directory under an output directory that stores per-config finalize shards.
pub const FINALIZE_SHARD_DIR: &str = "_finalize_shards";
/// File name for the serialized grid-matrix slice inside a finalize shard.
const GRID_MATRIX_SLICE_FILE: &str = "grid_matrix.bincode.zst";
/// Zstandard compression level for the grid-matrix slice.
const GRID_MATRIX_COMPRESSION_LEVEL: i32 = 6;
/// File name for per-shard pathway similarity-sum scores.
const PATHWAY_SCORE_SHARD_FILE: &str = "pathway_scores.parquet";
/// File name for per-shard pathway best-label predictions.
const PATHWAY_PREDICTION_SHARD_FILE: &str = "pathway_predictions.parquet";

/// Return whether both pathway shard files exist for a config and peak count.
#[must_use]
pub fn pathway_shard_exists(output_dir: &Path, config_name: &str, peak_count: usize) -> bool {
    pathway_shard_score_path(output_dir, config_name, peak_count).exists()
        && pathway_shard_prediction_path(output_dir, config_name, peak_count).exists()
}

/// Write pathway score and prediction shard files for one config/peak run.
///
/// # Errors
///
/// Returns an error when the shard directory cannot be created, either Parquet
/// file cannot be written, or a temporary file cannot be renamed into place.
pub fn write_pathway_shard(
    output_dir: &Path,
    config_name: &str,
    peak_count: usize,
    scores: &[PathwayScore],
    predictions: &[PathwayPrediction],
) -> Result<()> {
    let shard_dir = pathway_shard_dir(output_dir, config_name, peak_count);
    fs::create_dir_all(&shard_dir).with_context(|| format!("creating {}", shard_dir.display()))?;
    write_pathway_score_file(
        &pathway_shard_score_path(output_dir, config_name, peak_count),
        scores,
    )?;
    write_pathway_prediction_file(
        &pathway_shard_prediction_path(output_dir, config_name, peak_count),
        predictions,
    )
}

/// Return the pathway score shard path for one config and peak count.
fn pathway_shard_score_path(output_dir: &Path, config_name: &str, peak_count: usize) -> PathBuf {
    pathway_shard_dir(output_dir, config_name, peak_count).join(PATHWAY_SCORE_SHARD_FILE)
}

/// Return the pathway prediction shard path for one config and peak count.
fn pathway_shard_prediction_path(
    output_dir: &Path,
    config_name: &str,
    peak_count: usize,
) -> PathBuf {
    pathway_shard_dir(output_dir, config_name, peak_count).join(PATHWAY_PREDICTION_SHARD_FILE)
}

/// Return the shard directory for one config and peak count.
fn pathway_shard_dir(output_dir: &Path, config_name: &str, peak_count: usize) -> PathBuf {
    output_dir
        .join(PATHWAY_SHARD_DIR)
        .join(config_name)
        .join(format!("top_{peak_count:03}"))
}

/// Writers for all scan artifacts.
pub struct OutputWriters {
    /// Output directory used for late-written dense matrix artifacts.
    output_dir: PathBuf,
    /// Distribution summary output writer.
    summary: ParquetTableWriter,
    /// Histogram output writer.
    histogram: ParquetTableWriter,
    /// Adjacent-comparison output writer.
    adjacent_comparison: ParquetTableWriter,
    /// Full comparison-grid output writer.
    grid_comparison: ParquetTableWriter,
    /// Pathway score output writer.
    pathway_score: ParquetTableWriter,
    /// Pathway prediction output writer.
    pathway_prediction: ParquetTableWriter,
    /// Pending adjacent-comparison rows buffered into one row group per config.
    pending_adjacent_comparisons: Vec<DistributionComparison>,
    /// Pending grid rows buffered into one row group per config.
    pending_grid_comparisons: Vec<DistributionComparison>,
    /// Dense grid matrices written as `NumPy` arrays at the end of the run.
    grid_matrices: GridMatrixBuffer,
}

/// Layout in which an `OutputWriters` instance was finished.
#[derive(Clone, Copy)]
pub enum FinishMode<'a> {
    /// Write `distribution_grid.npz`, `distribution_grid_configs.parquet`,
    /// and heatmaps next to the per-config Parquet files.
    Aggregate,
    /// Save a per-config grid-matrix slice in the shard directory instead of
    /// writing the aggregate npz/configs, and render heatmaps under the
    /// canonical output directory rather than the shard subdirectory.
    PerConfigShard {
        /// Canonical (non-shard) output directory where heatmaps should land.
        canonical_dir: &'a Path,
    },
}

impl OutputWriters {
    /// Create every output writer under the scan output directory.
    pub fn create(output_dir: &Path) -> Result<Self> {
        Ok(Self {
            output_dir: output_dir.to_path_buf(),
            summary: ParquetTableWriter::create(
                output_dir,
                "distribution_summary.parquet",
                distribution_summary_schema(),
            )?,
            histogram: ParquetTableWriter::create(
                output_dir,
                "distribution_histograms.parquet",
                histogram_schema(),
            )?,
            adjacent_comparison: ParquetTableWriter::create(
                output_dir,
                "distribution_tests.parquet",
                distribution_comparison_schema(),
            )?,
            grid_comparison: ParquetTableWriter::create(
                output_dir,
                "distribution_grid.parquet",
                distribution_comparison_schema(),
            )?,
            pathway_score: ParquetTableWriter::create(
                output_dir,
                "pathway_scores.parquet",
                pathway_score_schema(),
            )?,
            pathway_prediction: ParquetTableWriter::create(
                output_dir,
                "pathway_predictions.parquet",
                pathway_prediction_schema(),
            )?,
            pending_adjacent_comparisons: Vec::new(),
            pending_grid_comparisons: Vec::new(),
            grid_matrices: GridMatrixBuffer::default(),
        })
    }

    /// Create writers under `<output_dir>/_finalize_shards/<config_name>/` for
    /// the per-config-shard finalize path.
    ///
    /// # Errors
    ///
    /// Returns an error if the shard directory cannot be created or any
    /// Parquet writer cannot be opened.
    pub fn create_for_shard(output_dir: &Path, config_name: &str) -> Result<Self> {
        let shard_dir = shard_directory(output_dir, config_name);
        fs::create_dir_all(&shard_dir)
            .with_context(|| format!("creating {}", shard_dir.display()))?;
        Self::create(&shard_dir)
    }

    /// Write one distribution summary row.
    pub fn write_summary(&mut self, summary: &DistributionSummary) -> Result<()> {
        self.summary
            .write(&distribution_summary_batch(std::slice::from_ref(summary))?)
    }

    /// Write histogram rows for one distribution.
    pub fn write_histogram(&mut self, bins: &[DistributionHistogramBin]) -> Result<()> {
        self.histogram.write(&histogram_batch(bins)?)
    }

    /// Write pathway score rows for one peak-count run.
    pub fn write_pathway_scores(&mut self, scores: &[PathwayScore]) -> Result<()> {
        self.pathway_score.write(&pathway_score_batch(scores)?)
    }

    /// Write pathway prediction rows for one peak-count run.
    pub fn write_pathway_predictions(&mut self, predictions: &[PathwayPrediction]) -> Result<()> {
        self.pathway_prediction
            .write(&pathway_prediction_batch(predictions)?)
    }

    /// Append pathway score and prediction rows from one completed shard.
    pub fn write_pathway_shard(
        &mut self,
        output_dir: &Path,
        config_name: &str,
        peak_count: usize,
    ) -> Result<()> {
        let score_path = pathway_shard_score_path(output_dir, config_name, peak_count);
        let prediction_path = pathway_shard_prediction_path(output_dir, config_name, peak_count);
        write_parquet_file_into_writer(&score_path, &mut self.pathway_score)?;
        write_parquet_file_into_writer(&prediction_path, &mut self.pathway_prediction)
    }

    /// Buffer one adjacent-distribution comparison.
    pub fn write_adjacent_comparison(&mut self, comparison: DistributionComparison) {
        self.pending_adjacent_comparisons.push(comparison);
    }

    /// Flush buffered adjacent-distribution comparisons.
    pub fn flush_adjacent_comparisons(&mut self) -> Result<()> {
        let rows = std::mem::take(&mut self.pending_adjacent_comparisons);
        self.adjacent_comparison
            .write(&distribution_comparison_batch(&rows)?)
    }

    /// Buffer one full-grid distribution comparison.
    pub fn write_grid_comparison(&mut self, comparison: DistributionComparison) {
        self.pending_grid_comparisons.push(comparison);
    }

    /// Flush buffered full-grid distribution comparisons.
    pub fn flush_grid_comparisons(&mut self) -> Result<()> {
        let rows = std::mem::take(&mut self.pending_grid_comparisons);
        for row in &rows {
            self.grid_matrices.push(row)?;
        }
        self.grid_comparison
            .write(&distribution_comparison_batch(&rows)?)
    }

    /// Finalize all Parquet files and write dense grid matrices.
    ///
    /// Convenience wrapper around `finish_with_mode` for the single-process
    /// (aggregate) finalize path.
    ///
    /// # Errors
    ///
    /// See `finish_with_mode`.
    pub fn finish(self, progress: &ScanProgress) -> Result<()> {
        self.finish_with_mode(FinishMode::Aggregate, progress)
    }

    /// Finalize all Parquet files in the configured mode.
    ///
    /// Parquet writers are closed before the heatmap renderer runs so a
    /// rendering failure (for example, a missing system font on a shared
    /// cluster) cannot leave the per-config Parquet artifacts truncated at
    /// their 4-byte header.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing buffered rows, building dense arrays,
    /// closing any Parquet writer, saving the grid-matrix slice, or rendering
    /// heatmaps fails.
    pub fn finish_with_mode(mut self, mode: FinishMode<'_>, progress: &ScanProgress) -> Result<()> {
        let adjacent_progress = progress.spinner("flushing adjacent comparison rows");
        self.flush_adjacent_comparisons()?;
        adjacent_progress.finish();

        let grid_progress = progress.spinner("flushing full-grid comparison rows");
        self.flush_grid_comparisons()?;
        grid_progress.finish();

        let matrix_progress = progress.spinner("building dense distribution matrices");
        let arrays = build_grid_arrays(&self.grid_matrices)?;
        matrix_progress.finish();

        match mode {
            FinishMode::Aggregate => {
                let npz_progress = progress.spinner("writing distribution_grid.npz");
                write_grid_npz(&self.output_dir, &arrays)?;
                npz_progress.finish();

                let config_progress = progress.spinner("writing distribution_grid_configs.parquet");
                write_grid_configs(&self.output_dir, &self.grid_matrices)?;
                config_progress.finish();
            }
            FinishMode::PerConfigShard { .. } => {
                let slice_progress = progress.spinner("saving grid-matrix shard slice");
                save_grid_matrix_slice(&self.output_dir, &self.grid_matrices)?;
                slice_progress.finish();
            }
        }

        let close_progress = progress.spinner("closing Parquet writers");
        self.summary.close()?;
        self.histogram.close()?;
        self.adjacent_comparison.close()?;
        self.grid_comparison.close()?;
        self.pathway_score.close()?;
        self.pathway_prediction.close()?;
        close_progress.finish();

        let heatmap_dir = match mode {
            FinishMode::Aggregate => self.output_dir.as_path(),
            FinishMode::PerConfigShard { canonical_dir } => canonical_dir,
        };
        write_heatmaps(heatmap_dir, &self.grid_matrices.configs, &arrays, progress)
    }
}

/// Streaming writer for one Parquet artifact.
struct ParquetTableWriter {
    /// Underlying Arrow-to-Parquet writer.
    writer: ArrowWriter<fs::File>,
}

impl ParquetTableWriter {
    /// Create one named Parquet writer under an output directory.
    fn create(output_dir: &Path, file_name: &str, schema: SchemaRef) -> Result<Self> {
        let path = output_dir.join(file_name);
        Self::create_at_path(&path, schema)
    }

    /// Create one Parquet writer at an explicit path.
    fn create_at_path(path: &Path, schema: SchemaRef) -> Result<Self> {
        let file =
            fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let writer = ArrowWriter::try_new(file, schema, None)
            .with_context(|| format!("opening Parquet writer for {}", path.display()))?;
        Ok(Self { writer })
    }

    /// Write one record batch when it contains rows.
    fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.num_rows() > 0 {
            self.writer.write(batch)?;
        }
        Ok(())
    }

    /// Close the writer and write the Parquet footer.
    fn close(self) -> Result<()> {
        self.writer.close()?;
        Ok(())
    }
}

/// Write one pathway score shard file atomically.
fn write_pathway_score_file(path: &Path, scores: &[PathwayScore]) -> Result<()> {
    let temporary_path = temporary_parquet_path(path);
    let mut writer = ParquetTableWriter::create_at_path(&temporary_path, pathway_score_schema())?;
    writer.write(&pathway_score_batch(scores)?)?;
    writer.close()?;
    rename_parquet(&temporary_path, path)
        .with_context(|| format!("moving {} to {}", temporary_path.display(), path.display()))
}

/// Write one pathway prediction shard file atomically.
fn write_pathway_prediction_file(path: &Path, predictions: &[PathwayPrediction]) -> Result<()> {
    let temporary_path = temporary_parquet_path(path);
    let mut writer =
        ParquetTableWriter::create_at_path(&temporary_path, pathway_prediction_schema())?;
    writer.write(&pathway_prediction_batch(predictions)?)?;
    writer.close()?;
    rename_parquet(&temporary_path, path)
        .with_context(|| format!("moving {} to {}", temporary_path.display(), path.display()))
}

/// Append every row group from a shard Parquet file into an open table writer.
fn write_parquet_file_into_writer(path: &Path, writer: &mut ParquetTableWriter) -> Result<()> {
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading metadata from {}", path.display()))?
        .build()
        .with_context(|| format!("building reader for {}", path.display()))?;
    for batch in reader {
        writer.write(&batch.with_context(|| format!("reading batch from {}", path.display()))?)?;
    }
    Ok(())
}

/// Return a process-specific temporary Parquet path next to the final path.
fn temporary_parquet_path(path: &Path) -> PathBuf {
    let file_name = path.file_name().map_or_else(
        || "artifact.parquet".into(),
        |file_name| file_name.to_string_lossy(),
    );
    path.with_file_name(format!("{file_name}.tmp-{}", std::process::id()))
}

/// Rename a temporary Parquet artifact into place, replacing any previous file.
fn rename_parquet(temporary_path: &Path, path: &Path) -> std::io::Result<()> {
    match fs::rename(temporary_path, path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {
            fs::remove_file(path)?;
            fs::rename(temporary_path, path)
        }
        Err(error) => Err(error),
    }
}

/// Dense comparison grid arrays accumulated across similarity configs.
#[derive(Default, Serialize, Deserialize)]
struct GridMatrixBuffer {
    /// Similarity configuration labels in first-seen order.
    configs: Vec<String>,
    /// Metric labels aligned to `configs`.
    metrics: Vec<String>,
    /// Buffered `(config, row, column, comparison)` entries.
    entries: Vec<GridMatrixEntry>,
}

impl GridMatrixBuffer {
    /// Add one comparison to the dense matrix buffer.
    fn push(&mut self, comparison: &DistributionComparison) -> Result<()> {
        let config_index = self.config_index(comparison);
        let row = matrix_index(comparison.peak_count_a)?;
        let column = matrix_index(comparison.peak_count_b)?;
        self.entries.push(GridMatrixEntry {
            config_index,
            row,
            column,
            mean_delta: comparison.mean_delta,
            ks_statistic: comparison.ks_statistic,
            ks_pvalue_asymptotic: comparison.ks_pvalue_asymptotic,
            wasserstein_1d: comparison.wasserstein_1d,
        });
        Ok(())
    }

    /// Return the config index, inserting a new config when needed.
    fn config_index(&mut self, comparison: &DistributionComparison) -> usize {
        if let Some(index) = self
            .configs
            .iter()
            .position(|config| config == &comparison.config)
        {
            return index;
        }
        self.configs.push(comparison.config.clone());
        self.metrics.push(comparison.metric.to_string());
        self.configs.len() - 1
    }

    /// Append all entries from another buffer, reindexing their `config_index`
    /// to point at this buffer's existing or freshly-inserted slot for the
    /// other buffer's single config.
    ///
    /// The other buffer must describe exactly one config — that's what
    /// per-config shards produce. Any wider buffer is rejected.
    fn extend_from_shard(&mut self, other: Self) -> Result<()> {
        if other.configs.len() != 1 {
            bail!(
                "expected a one-config grid-matrix slice, got {} configs",
                other.configs.len()
            );
        }
        let canonical_index = self.configs.len();
        self.configs.push(other.configs[0].clone());
        self.metrics.push(other.metrics[0].clone());
        for entry in other.entries {
            self.entries.push(GridMatrixEntry {
                config_index: canonical_index,
                row: entry.row,
                column: entry.column,
                mean_delta: entry.mean_delta,
                ks_statistic: entry.ks_statistic,
                ks_pvalue_asymptotic: entry.ks_pvalue_asymptotic,
                wasserstein_1d: entry.wasserstein_1d,
            });
        }
        Ok(())
    }
}

/// Dense comparison matrices aligned to the config axis.
pub struct GridArrays {
    /// One-based retained peak-count axis.
    pub peak_counts: Array1<u64>,
    /// Mean-score delta matrix.
    pub mean_delta: Array3<f64>,
    /// Kolmogorov-Smirnov statistic matrix.
    pub ks_statistic: Array3<f64>,
    /// Asymptotic Kolmogorov-Smirnov p-value matrix.
    pub ks_pvalue_asymptotic: Array3<f64>,
    /// One-dimensional Wasserstein-distance matrix.
    pub wasserstein_1d: Array3<f64>,
}

/// One dense matrix cell.
#[derive(Serialize, Deserialize)]
struct GridMatrixEntry {
    /// First axis index for the similarity configuration.
    config_index: usize,
    /// Second axis index for `peak_count_a`.
    row: usize,
    /// Third axis index for `peak_count_b`.
    column: usize,
    /// Mean-score delta for the comparison.
    mean_delta: f64,
    /// Kolmogorov-Smirnov statistic for the comparison.
    ks_statistic: f64,
    /// Asymptotic Kolmogorov-Smirnov p-value approximation.
    ks_pvalue_asymptotic: f64,
    /// One-dimensional Wasserstein distance for the comparison.
    wasserstein_1d: f64,
}

/// Build dense full-grid arrays from buffered comparison rows.
fn build_grid_arrays(buffer: &GridMatrixBuffer) -> Result<GridArrays> {
    let shape = (
        buffer.configs.len(),
        PEAK_COUNT_GRID_SIZE,
        PEAK_COUNT_GRID_SIZE,
    );
    let mut mean_delta = Array3::<f64>::zeros(shape);
    let mut ks_statistic = Array3::<f64>::zeros(shape);
    let mut ks_pvalue_asymptotic = Array3::<f64>::zeros(shape);
    let mut wasserstein_1d = Array3::<f64>::zeros(shape);

    for entry in &buffer.entries {
        let index = (entry.config_index, entry.row, entry.column);
        mean_delta[index] = entry.mean_delta;
        ks_statistic[index] = entry.ks_statistic;
        ks_pvalue_asymptotic[index] = entry.ks_pvalue_asymptotic;
        wasserstein_1d[index] = entry.wasserstein_1d;
    }

    Ok(GridArrays {
        peak_counts: peak_counts_array()?,
        mean_delta,
        ks_statistic,
        ks_pvalue_asymptotic,
        wasserstein_1d,
    })
}

/// Write dense full-grid arrays into `distribution_grid.npz`.
fn write_grid_npz(output_dir: &Path, arrays: &GridArrays) -> Result<()> {
    let path = output_dir.join("distribution_grid.npz");
    let file = fs::File::create(&path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = NpzWriter::new(file);
    writer.add_array("peak_counts", &arrays.peak_counts)?;
    writer.add_array("mean_delta", &arrays.mean_delta)?;
    writer.add_array("ks_statistic", &arrays.ks_statistic)?;
    writer.add_array("ks_pvalue_asymptotic", &arrays.ks_pvalue_asymptotic)?;
    writer.add_array("wasserstein_1d", &arrays.wasserstein_1d)?;
    writer.finish()?;
    Ok(())
}

/// Return the per-config shard subdirectory for a finalize shard.
#[must_use]
pub fn shard_directory(output_dir: &Path, config_name: &str) -> PathBuf {
    output_dir.join(FINALIZE_SHARD_DIR).join(config_name)
}

/// Return the canonical path to a per-config grid-matrix slice file.
#[must_use]
pub fn grid_matrix_slice_path(output_dir: &Path, config_name: &str) -> PathBuf {
    shard_directory(output_dir, config_name).join(GRID_MATRIX_SLICE_FILE)
}

/// Serialize a grid-matrix buffer to a per-config slice file.
fn save_grid_matrix_slice(shard_dir: &Path, buffer: &GridMatrixBuffer) -> Result<()> {
    use std::io::Write;
    let path = shard_dir.join(GRID_MATRIX_SLICE_FILE);
    let bytes = bincode::serialize(buffer)
        .with_context(|| format!("encoding grid-matrix slice for {}", path.display()))?;
    let file = fs::File::create(&path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = zstd::Encoder::new(file, GRID_MATRIX_COMPRESSION_LEVEL)
        .with_context(|| format!("opening zstd encoder for {}", path.display()))?;
    writer
        .write_all(&bytes)
        .with_context(|| format!("writing {}", path.display()))?;
    writer
        .finish()
        .with_context(|| format!("finalizing zstd stream for {}", path.display()))?;
    Ok(())
}

/// Reconstitute a per-config grid-matrix slice from disk.
fn load_grid_matrix_slice(path: &Path) -> Result<GridMatrixBuffer> {
    use std::io::Read;
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut decoder = zstd::Decoder::new(file)
        .with_context(|| format!("opening zstd decoder for {}", path.display()))?;
    let mut bytes = Vec::new();
    decoder
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading {}", path.display()))?;
    let buffer: GridMatrixBuffer = bincode::deserialize(&bytes)
        .with_context(|| format!("decoding grid-matrix slice {}", path.display()))?;
    Ok(buffer)
}

/// Stack per-config grid-matrix slices into the canonical aggregate npz and
/// `distribution_grid_configs.parquet` files.
///
/// # Errors
///
/// Returns an error if any slice fails to load, contains the wrong number of
/// configs, or the canonical npz/configs files cannot be written.
pub fn merge_grid_matrix_slices(output_dir: &Path, config_names: &[String]) -> Result<()> {
    let mut combined = GridMatrixBuffer::default();
    for config_name in config_names {
        let slice_path = grid_matrix_slice_path(output_dir, config_name);
        let slice = load_grid_matrix_slice(&slice_path)?;
        combined.extend_from_shard(slice)?;
    }
    let arrays = build_grid_arrays(&combined)?;
    write_grid_npz(output_dir, &arrays)?;
    write_grid_configs(output_dir, &combined)?;
    Ok(())
}

/// Concatenate each shard's per-config Parquets into the canonical aggregate
/// files.
///
/// The shard layout `<output_dir>/_finalize_shards/<config>/<file>.parquet` is
/// expected. Pathway Parquets are concatenated only when pathway scoring was
/// enabled for the run.
///
/// # Errors
///
/// Returns an error if any source Parquet cannot be read or any destination
/// cannot be written.
pub fn merge_per_config_parquets(
    output_dir: &Path,
    config_names: &[String],
    include_pathway_outputs: bool,
) -> Result<()> {
    let sources_for = |file_name: &str| -> Vec<PathBuf> {
        config_names
            .iter()
            .map(|config| shard_directory(output_dir, config).join(file_name))
            .collect()
    };
    let merge_one = |file_name: &str, schema: SchemaRef| -> Result<()> {
        concat_parquet(&sources_for(file_name), &output_dir.join(file_name), schema)
    };

    merge_one(
        "distribution_summary.parquet",
        distribution_summary_schema(),
    )?;
    merge_one("distribution_histograms.parquet", histogram_schema())?;
    merge_one(
        "distribution_tests.parquet",
        distribution_comparison_schema(),
    )?;
    merge_one(
        "distribution_grid.parquet",
        distribution_comparison_schema(),
    )?;
    if include_pathway_outputs {
        merge_one("pathway_scores.parquet", pathway_score_schema())?;
        merge_one("pathway_predictions.parquet", pathway_prediction_schema())?;
    }
    Ok(())
}

/// Concatenate a sequence of Parquet files with the same schema into one.
///
/// Used by `finalize-merge` to stitch per-config shard Parquets into the
/// canonical aggregate file. Empty input files (just the header) and missing
/// files are tolerated.
///
/// # Errors
///
/// Returns an error if any source file fails to open or any batch cannot be
/// written into the destination writer.
pub fn concat_parquet(sources: &[PathBuf], destination: &Path, schema: SchemaRef) -> Result<()> {
    let mut writer = ParquetTableWriter::create_at_path(destination, schema)?;
    for source in sources {
        if !source.is_file() {
            continue;
        }
        let file =
            fs::File::open(source).with_context(|| format!("opening {}", source.display()))?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("reading metadata from {}", source.display()))?
            .build()
            .with_context(|| format!("building reader for {}", source.display()))?;
        for batch in reader {
            let batch =
                batch.with_context(|| format!("reading row group from {}", source.display()))?;
            writer.write(&batch)?;
        }
    }
    writer.close()
}

/// Write config-axis metadata for `distribution_grid.npz`.
fn write_grid_configs(output_dir: &Path, buffer: &GridMatrixBuffer) -> Result<()> {
    let mut rows = Vec::with_capacity(buffer.configs.len());
    for (config_index, (config, metric)) in buffer.configs.iter().zip(&buffer.metrics).enumerate() {
        rows.push(GridConfigRow {
            config_index,
            config: config.clone(),
            metric: metric.clone(),
        });
    }
    let mut writer = ParquetTableWriter::create(
        output_dir,
        "distribution_grid_configs.parquet",
        grid_config_schema(),
    )?;
    writer.write(&grid_config_batch(&rows)?)?;
    writer.close()
}

/// Return the zero-based matrix index for a one-based peak count.
fn matrix_index(peak_count: usize) -> Result<usize> {
    let index = peak_count.checked_sub(1).with_context(|| {
        format!("peak count {peak_count} cannot be represented as a matrix index")
    })?;
    if index >= PEAK_COUNT_GRID_SIZE {
        bail!("peak count {peak_count} exceeds dense grid size {PEAK_COUNT_GRID_SIZE}");
    }
    Ok(index)
}

/// Return the `1..=128` peak-count axis as a `NumPy` array.
fn peak_counts_array() -> Result<Array1<u64>> {
    let values = (1..=PEAK_COUNT_GRID_SIZE)
        .map(usize_to_u64)
        .collect::<Result<Vec<_>>>()?;
    Ok(Array1::from(values))
}

/// Config metadata row for the dense grid axis.
struct GridConfigRow {
    /// Zero-based config axis index.
    config_index: usize,
    /// Similarity configuration label.
    config: String,
    /// Similarity metric label.
    metric: String,
}

/// Build a Parquet batch for distribution summaries.
fn distribution_summary_batch(rows: &[DistributionSummary]) -> Result<RecordBatch> {
    let columns = vec![
        strings(
            rows.iter()
                .map(|row| row.dataset.as_str())
                .collect::<Vec<_>>(),
        ),
        strings(
            rows.iter()
                .map(|row| row.config.as_str())
                .collect::<Vec<_>>(),
        ),
        strings(rows.iter().map(|row| row.metric).collect::<Vec<_>>()),
        usizes(rows.iter().map(|row| row.peak_count).collect::<Vec<_>>())?,
        usizes(rows.iter().map(|row| row.n_scores).collect::<Vec<_>>())?,
        floats(rows.iter().map(|row| row.mean).collect::<Vec<_>>()),
        floats(rows.iter().map(|row| row.stddev).collect::<Vec<_>>()),
        floats(rows.iter().map(|row| row.min).collect::<Vec<_>>()),
        floats(rows.iter().map(|row| row.q01).collect::<Vec<_>>()),
        floats(rows.iter().map(|row| row.q05).collect::<Vec<_>>()),
        floats(rows.iter().map(|row| row.q10).collect::<Vec<_>>()),
        floats(rows.iter().map(|row| row.q25).collect::<Vec<_>>()),
        floats(rows.iter().map(|row| row.median).collect::<Vec<_>>()),
        floats(rows.iter().map(|row| row.q75).collect::<Vec<_>>()),
        floats(rows.iter().map(|row| row.q90).collect::<Vec<_>>()),
        floats(rows.iter().map(|row| row.q95).collect::<Vec<_>>()),
        floats(rows.iter().map(|row| row.q99).collect::<Vec<_>>()),
        floats(rows.iter().map(|row| row.max).collect::<Vec<_>>()),
    ];
    record_batch(
        distribution_summary_schema(),
        columns,
        "distribution summaries",
    )
}

/// Build a Parquet batch for histogram bins.
fn histogram_batch(rows: &[DistributionHistogramBin]) -> Result<RecordBatch> {
    let columns = vec![
        strings(
            rows.iter()
                .map(|row| row.dataset.as_str())
                .collect::<Vec<_>>(),
        ),
        strings(
            rows.iter()
                .map(|row| row.config.as_str())
                .collect::<Vec<_>>(),
        ),
        strings(rows.iter().map(|row| row.metric).collect::<Vec<_>>()),
        usizes(rows.iter().map(|row| row.peak_count).collect::<Vec<_>>())?,
        usizes(rows.iter().map(|row| row.bin_index).collect::<Vec<_>>())?,
        floats(rows.iter().map(|row| row.bin_lower).collect::<Vec<_>>()),
        floats(rows.iter().map(|row| row.bin_upper).collect::<Vec<_>>()),
        usizes(rows.iter().map(|row| row.count).collect::<Vec<_>>())?,
        floats(rows.iter().map(|row| row.fraction).collect::<Vec<_>>()),
    ];
    record_batch(histogram_schema(), columns, "histogram bins")
}

/// Build a Parquet batch for distribution comparisons.
fn distribution_comparison_batch(rows: &[DistributionComparison]) -> Result<RecordBatch> {
    let columns = vec![
        strings(
            rows.iter()
                .map(|row| row.dataset.as_str())
                .collect::<Vec<_>>(),
        ),
        strings(
            rows.iter()
                .map(|row| row.config.as_str())
                .collect::<Vec<_>>(),
        ),
        strings(rows.iter().map(|row| row.metric).collect::<Vec<_>>()),
        usizes(rows.iter().map(|row| row.peak_count_a).collect::<Vec<_>>())?,
        usizes(rows.iter().map(|row| row.peak_count_b).collect::<Vec<_>>())?,
        usizes(rows.iter().map(|row| row.n_scores_a).collect::<Vec<_>>())?,
        usizes(rows.iter().map(|row| row.n_scores_b).collect::<Vec<_>>())?,
        floats(rows.iter().map(|row| row.mean_a).collect::<Vec<_>>()),
        floats(rows.iter().map(|row| row.mean_b).collect::<Vec<_>>()),
        floats(rows.iter().map(|row| row.mean_delta).collect::<Vec<_>>()),
        floats(rows.iter().map(|row| row.ks_statistic).collect::<Vec<_>>()),
        floats(
            rows.iter()
                .map(|row| row.ks_pvalue_asymptotic)
                .collect::<Vec<_>>(),
        ),
        floats(
            rows.iter()
                .map(|row| row.wasserstein_1d)
                .collect::<Vec<_>>(),
        ),
    ];
    record_batch(
        distribution_comparison_schema(),
        columns,
        "distribution comparisons",
    )
}

/// Build a Parquet batch for pathway scores.
fn pathway_score_batch(rows: &[PathwayScore]) -> Result<RecordBatch> {
    let columns = vec![
        strings(
            rows.iter()
                .map(|row| row.dataset.as_str())
                .collect::<Vec<_>>(),
        ),
        strings(
            rows.iter()
                .map(|row| row.config.as_str())
                .collect::<Vec<_>>(),
        ),
        usizes(rows.iter().map(|row| row.peak_count).collect::<Vec<_>>())?,
        usizes(rows.iter().map(|row| row.query_index).collect::<Vec<_>>())?,
        strings(
            rows.iter()
                .map(|row| row.query_id.as_str())
                .collect::<Vec<_>>(),
        ),
        optional_strings(
            rows.iter()
                .map(|row| row.query_npc_pathway.as_deref())
                .collect::<Vec<_>>(),
        ),
        strings(
            rows.iter()
                .map(|row| row.candidate_npc_pathway.as_str())
                .collect::<Vec<_>>(),
        ),
        usizes(
            rows.iter()
                .map(|row| row.representatives)
                .collect::<Vec<_>>(),
        )?,
        floats(rows.iter().map(|row| row.score).collect::<Vec<_>>()),
    ];
    record_batch(pathway_score_schema(), columns, "pathway scores")
}

/// Build a Parquet batch for pathway predictions.
fn pathway_prediction_batch(rows: &[PathwayPrediction]) -> Result<RecordBatch> {
    let columns = vec![
        strings(
            rows.iter()
                .map(|row| row.dataset.as_str())
                .collect::<Vec<_>>(),
        ),
        strings(
            rows.iter()
                .map(|row| row.config.as_str())
                .collect::<Vec<_>>(),
        ),
        usizes(rows.iter().map(|row| row.peak_count).collect::<Vec<_>>())?,
        usizes(rows.iter().map(|row| row.query_index).collect::<Vec<_>>())?,
        strings(
            rows.iter()
                .map(|row| row.query_id.as_str())
                .collect::<Vec<_>>(),
        ),
        optional_strings(
            rows.iter()
                .map(|row| row.query_npc_pathway.as_deref())
                .collect::<Vec<_>>(),
        ),
        optional_strings(
            rows.iter()
                .map(|row| row.predicted_npc_pathway.as_deref())
                .collect::<Vec<_>>(),
        ),
        floats(
            rows.iter()
                .map(|row| row.predicted_score)
                .collect::<Vec<_>>(),
        ),
        optional_booleans(rows.iter().map(|row| row.is_correct).collect::<Vec<_>>()),
        usizes(
            rows.iter()
                .map(|row| row.candidate_pathways)
                .collect::<Vec<_>>(),
        )?,
    ];
    record_batch(pathway_prediction_schema(), columns, "pathway predictions")
}

/// Build a Parquet batch for dense grid config metadata.
fn grid_config_batch(rows: &[GridConfigRow]) -> Result<RecordBatch> {
    let columns = vec![
        usizes(rows.iter().map(|row| row.config_index).collect::<Vec<_>>())?,
        strings(
            rows.iter()
                .map(|row| row.config.as_str())
                .collect::<Vec<_>>(),
        ),
        strings(
            rows.iter()
                .map(|row| row.metric.as_str())
                .collect::<Vec<_>>(),
        ),
    ];
    record_batch(grid_config_schema(), columns, "grid configs")
}

/// Build a record batch with contextual errors.
fn record_batch(schema: SchemaRef, columns: Vec<ArrayRef>, label: &str) -> Result<RecordBatch> {
    RecordBatch::try_new(schema, columns).with_context(|| format!("building {label} batch"))
}

/// Return the distribution summary output schema.
fn distribution_summary_schema() -> SchemaRef {
    schema(vec![
        utf8("dataset", false),
        utf8("config", false),
        utf8("metric", false),
        u64_field("peak_count", false),
        u64_field("n_scores", false),
        f64_field("mean", false),
        f64_field("stddev", false),
        f64_field("min", false),
        f64_field("q01", false),
        f64_field("q05", false),
        f64_field("q10", false),
        f64_field("q25", false),
        f64_field("median", false),
        f64_field("q75", false),
        f64_field("q90", false),
        f64_field("q95", false),
        f64_field("q99", false),
        f64_field("max", false),
    ])
}

/// Return the histogram output schema.
fn histogram_schema() -> SchemaRef {
    schema(vec![
        utf8("dataset", false),
        utf8("config", false),
        utf8("metric", false),
        u64_field("peak_count", false),
        u64_field("bin_index", false),
        f64_field("bin_lower", false),
        f64_field("bin_upper", false),
        u64_field("count", false),
        f64_field("fraction", false),
    ])
}

/// Return the distribution comparison output schema.
fn distribution_comparison_schema() -> SchemaRef {
    schema(vec![
        utf8("dataset", false),
        utf8("config", false),
        utf8("metric", false),
        u64_field("peak_count_a", false),
        u64_field("peak_count_b", false),
        u64_field("n_scores_a", false),
        u64_field("n_scores_b", false),
        f64_field("mean_a", false),
        f64_field("mean_b", false),
        f64_field("mean_delta", false),
        f64_field("ks_statistic", false),
        f64_field("ks_pvalue_asymptotic", false),
        f64_field("wasserstein_1d", false),
    ])
}

/// Return the pathway score output schema.
fn pathway_score_schema() -> SchemaRef {
    schema(vec![
        utf8("dataset", false),
        utf8("config", false),
        u64_field("peak_count", false),
        u64_field("query_index", false),
        utf8("query_id", false),
        utf8("query_npc_pathway", true),
        utf8("candidate_npc_pathway", false),
        u64_field("representatives", false),
        f64_field("score", false),
    ])
}

/// Return the pathway prediction output schema.
fn pathway_prediction_schema() -> SchemaRef {
    schema(vec![
        utf8("dataset", false),
        utf8("config", false),
        u64_field("peak_count", false),
        u64_field("query_index", false),
        utf8("query_id", false),
        utf8("query_npc_pathway", true),
        utf8("predicted_npc_pathway", true),
        f64_field("predicted_score", false),
        bool_field("is_correct", true),
        u64_field("candidate_pathways", false),
    ])
}

/// Return the grid config metadata schema.
fn grid_config_schema() -> SchemaRef {
    schema(vec![
        u64_field("config_index", false),
        utf8("config", false),
        utf8("metric", false),
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

/// Build a nullable UTF-8 array.
fn optional_strings(values: Vec<Option<&str>>) -> ArrayRef {
    Arc::new(StringArray::from(values))
}

/// Build a required `f64` array.
fn floats(values: Vec<f64>) -> ArrayRef {
    Arc::new(Float64Array::from(values))
}

/// Build a nullable boolean array.
fn optional_booleans(values: Vec<Option<bool>>) -> ArrayRef {
    Arc::new(BooleanArray::from(values))
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

/// Convert a `usize` into `u64` with an explicit error path for portability.
fn usize_to_u64(value: usize) -> Result<u64> {
    u64::try_from(value).context("usize value does not fit u64")
}

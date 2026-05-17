//! Per-config AUROC and AUPRC of pathway-pair similarity scores.
//!
//! Reads `pathway_scores.parquet` (the merged file produced by
//! `finalize-scan`) when present; otherwise walks
//! `pathway_shards/<config>/top_<k>/pathway_scores.parquet` directly so we
//! can compute discriminability before merging a transferred shard tree.
//!
//! Each row is labelled by `candidate_npc_pathway == query_npc_pathway`.
//! For each `(dataset, config, peak_count, query_pathway)` group we compute
//! a per-class (one-vs-rest) AUROC and AUPRC, treating queries in that
//! pathway as the fixed positive class — emitted as
//! `pathway_discriminability_per_class.parquet`. The micro-averaged
//! aggregate over all queries (every same-pathway pair pooled against every
//! different-pathway pair) is then computed from the union of the per-class
//! samples and emitted as `pathway_discriminability.parquet` and its
//! per-config mean summary.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::missing_docs_in_private_items,
    clippy::type_complexity
)]

use std::cmp::Ordering;
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, bail};
use arrow_array::{Array, Float64Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use parquet::arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder};
use rayon::prelude::*;

use crate::output::parquet_writer_props;
use crate::progress::ScanProgress;

/// Compute the area under the ROC curve (AUROC) of a binary-labelled score
/// sample. Uses the Mann–Whitney U identity with average-rank tie
/// correction, matching scikit-learn's `roc_auc_score`.
///
/// Returns NaN when there are no positives or no negatives.
#[must_use]
pub fn auroc(samples: &mut [(f64, bool)]) -> f64 {
    let n_total = samples.len();
    if n_total == 0 {
        return f64::NAN;
    }
    let n_pos = samples.iter().filter(|(_, label)| *label).count();
    let n_neg = n_total - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return f64::NAN;
    }
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let mut sum_ranks_pos = 0.0_f64;
    let mut i = 0;
    while i < n_total {
        let mut j = i;
        while j < n_total && samples[j].0.partial_cmp(&samples[i].0) == Some(Ordering::Equal) {
            j += 1;
        }
        // Average rank for tied positions (1-indexed): ((i+1) + j) / 2.
        let avg_rank = ((i + 1) + j) as f64 / 2.0;
        for sample in &samples[i..j] {
            if sample.1 {
                sum_ranks_pos += avg_rank;
            }
        }
        i = j;
    }
    let half_n_pos_corr = (n_pos as f64) * ((n_pos as f64) + 1.0) / 2.0;
    (sum_ranks_pos - half_n_pos_corr) / ((n_pos as f64) * (n_neg as f64))
}

/// Compute the average precision (area under the Precision-Recall curve)
/// of a binary-labelled score sample. Matches scikit-learn's
/// `average_precision_score`, including its behavior on ties (within a tied
/// block the precision is averaged before incorporation, which is what
/// scikit-learn does via the trapezoid-like formulation).
///
/// Returns NaN when there are no positives in the sample.
#[must_use]
pub fn auprc(samples: &mut [(f64, bool)]) -> f64 {
    let n_pos = samples.iter().filter(|(_, label)| *label).count();
    if n_pos == 0 {
        return f64::NAN;
    }
    // Sort by score descending so the top-ranked is at index 0.
    samples.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

    let mut ap = 0.0_f64;
    let mut tp = 0_usize;
    let mut fp = 0_usize;
    let mut i = 0;
    while i < samples.len() {
        let mut j = i;
        // Walk through tied scores collectively to avoid order-dependent
        // ranking inside a tie block (matches scikit-learn semantics).
        let mut tied_pos = 0_usize;
        let mut tied_neg = 0_usize;
        while j < samples.len() && samples[j].0.partial_cmp(&samples[i].0) == Some(Ordering::Equal)
        {
            if samples[j].1 {
                tied_pos += 1;
            } else {
                tied_neg += 1;
            }
            j += 1;
        }
        tp += tied_pos;
        fp += tied_neg;
        if tied_pos > 0 {
            let precision = tp as f64 / (tp + fp) as f64;
            ap = precision.mul_add(tied_pos as f64 / n_pos as f64, ap);
        }
        i = j;
    }
    ap
}

/// Compute per-`(config, peak_count)` AUROC / AUPRC from
/// `pathway_scores.parquet` and persist both a per-cell table and a
/// per-config mean-summary table under `output_dir`.
///
/// Returns silently when the parquet is missing or has zero rows so the
/// command can be safely invoked on datasets without pathway scoring
/// enabled (e.g., `gems-sampled`).
///
/// # Errors
///
/// Returns an error when the input parquet exists but cannot be opened or
/// parsed, or when the output parquets cannot be written.
pub fn write_pathway_discriminability(
    output_dir: &Path,
    from_merged: bool,
    progress: &ScanProgress,
) -> Result<()> {
    let merged_path = output_dir.join("pathway_scores.parquet");
    let merged_size = fs::metadata(&merged_path).map_or(0, |m| m.len());
    let shard_root = output_dir.join("pathway_shards");

    // Prefer the per-shard tree when it exists. Each shard is exactly one
    // `(config, peak_count)` cell, so the streaming reader can compute
    // AUROC / AUPRC one shard at a time and drop the buffer; the merged
    // path holds every cell's scores in a single in-memory `HashMap`,
    // which would need hundreds of GB of RAM on the harmonized corpus
    // (the cluster OOM-killed an earlier run when this branch was
    // preferred). Fall back to the merged file only when no shard
    // directory is available — or when the caller forced
    // `--from-merged` because the local shard tree is partial.
    let shard_paths = if from_merged || !shard_root.is_dir() {
        Vec::new()
    } else {
        collect_pathway_shard_paths(&shard_root)?
    };

    let (mut cell_rows, mut per_class_rows) = if !shard_paths.is_empty() {
        let task = progress.bar(
            u64::try_from(shard_paths.len()).unwrap_or(u64::MAX),
            "computing AUROC / AUPRC per shard",
        );
        let rows = compute_rows_from_shards(&shard_paths, &task)?;
        task.finish();
        rows
    } else if merged_path.is_file() && merged_size >= 1024 {
        let read_progress = progress.spinner("reading pathway scores (merged)");
        let groups = read_pathway_scores_grouped(&merged_path)?;
        read_progress.finish();
        if groups.is_empty() {
            return Ok(());
        }
        let compute_progress = progress.spinner("computing AUROC / AUPRC per (config, peak_count)");
        let rows = compute_rows_from_groups(groups);
        compute_progress.finish();
        rows
    } else {
        return Ok(());
    };
    if cell_rows.is_empty() {
        return Ok(());
    }
    cell_rows.sort_by(|a, b| {
        a.dataset
            .cmp(&b.dataset)
            .then(a.config.cmp(&b.config))
            .then(a.peak_count.cmp(&b.peak_count))
    });
    per_class_rows.sort_by(|a, b| {
        a.dataset
            .cmp(&b.dataset)
            .then(a.pathway.cmp(&b.pathway))
            .then(a.config.cmp(&b.config))
            .then(a.peak_count.cmp(&b.peak_count))
    });

    let summary_rows = summarize_per_config(&cell_rows);

    let write_progress = progress.spinner("writing pathway_discriminability artifacts");
    write_cell_rows(output_dir, &cell_rows)?;
    write_summary_rows(output_dir, &summary_rows)?;
    write_per_class_rows(output_dir, &per_class_rows)?;
    write_progress.finish();
    Ok(())
}

/// One row of `pathway_discriminability.parquet`.
struct CellRow {
    dataset: String,
    config: String,
    peak_count: u64,
    auroc: f64,
    auprc: f64,
    n_positives: u64,
    n_negatives: u64,
}

/// One row of `pathway_discriminability_summary.parquet`.
struct SummaryRow {
    dataset: String,
    config: String,
    mean_auroc: f64,
    mean_auprc: f64,
    n_peak_counts: u64,
}

/// One row of `pathway_discriminability_per_class.parquet`. `pathway` is
/// the fixed positive class for the one-vs-rest classifier: positives are
/// candidate pairs whose `candidate_npc_pathway` equals `pathway`, drawn
/// from queries whose `query_npc_pathway` also equals `pathway`.
struct PerClassRow {
    dataset: String,
    config: String,
    peak_count: u64,
    pathway: String,
    auroc: f64,
    auprc: f64,
    n_positives: u64,
    n_negatives: u64,
}

/// Key used while accumulating pathway-pair scores: `(dataset, config,
/// peak_count, query_pathway)`. Aggregating over `query_pathway` recovers
/// the original `(dataset, config, peak_count)` grouping.
type PerClassKey = (String, String, u64, String);
/// Per-class score buckets used to compute one-vs-rest AUROC / AUPRC.
type PerClassGroups = HashMap<PerClassKey, Vec<(f64, bool)>>;

fn read_pathway_scores_grouped(path: &Path) -> Result<PerClassGroups> {
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading metadata from {}", path.display()))?
        .build()
        .with_context(|| format!("building reader for {}", path.display()))?;

    let mut groups: PerClassGroups = HashMap::new();
    for batch in reader {
        let batch = batch.with_context(|| format!("decoding batch in {}", path.display()))?;
        observe_batch(&batch, &mut groups)?;
    }
    Ok(groups)
}

/// Walk `pathway_shards/<config>/top_<k>/pathway_scores.parquet` and return
/// every shard file found. Sorted for deterministic progress reporting.
fn collect_pathway_shard_paths(shard_root: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    let config_entries =
        fs::read_dir(shard_root).with_context(|| format!("reading {}", shard_root.display()))?;
    for config_entry in config_entries {
        let config_entry =
            config_entry.with_context(|| format!("listing {}", shard_root.display()))?;
        if !config_entry.file_type()?.is_dir() {
            continue;
        }
        let peak_entries = fs::read_dir(config_entry.path())
            .with_context(|| format!("reading {}", config_entry.path().display()))?;
        for peak_entry in peak_entries {
            let peak_entry =
                peak_entry.with_context(|| format!("listing {}", config_entry.path().display()))?;
            if !peak_entry.file_type()?.is_dir() {
                continue;
            }
            let scores = peak_entry.path().join("pathway_scores.parquet");
            if scores.is_file() {
                paths.push(scores);
            }
        }
    }
    paths.sort();
    Ok(paths)
}

/// Stream every shard in `paths` in parallel, computing AUROC / AUPRC for
/// each shard's `(dataset, config, peak_count)` group (both per-pathway
/// one-vs-rest and the micro-pooled aggregate) and dropping the per-shard
/// buffer before moving on.
///
/// Each shard file is exactly one `(config, peak_count)` cell of pairwise
/// scores (one cell can hold tens of millions of rows), so loading them
/// all simultaneously is not viable at cluster scale. The streaming
/// per-shard approach keeps peak memory at roughly `rayon` thread count
/// times the largest cell's score buffer.
fn compute_rows_from_shards(
    paths: &[PathBuf],
    progress: &crate::progress::ProgressTask,
) -> Result<(Vec<CellRow>, Vec<PerClassRow>)> {
    paths
        .par_iter()
        .map(|path| {
            let mut groups: PerClassGroups = HashMap::new();
            let file =
                fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
            let reader = ParquetRecordBatchReaderBuilder::try_new(file)
                .with_context(|| format!("reading metadata from {}", path.display()))?
                .build()
                .with_context(|| format!("building reader for {}", path.display()))?;
            for batch in reader {
                let batch =
                    batch.with_context(|| format!("decoding batch in {}", path.display()))?;
                observe_batch(&batch, &mut groups)?;
            }
            let rows = compute_rows_from_groups(groups);
            progress.inc(1);
            Ok::<(Vec<CellRow>, Vec<PerClassRow>), anyhow::Error>(rows)
        })
        .reduce(
            || Ok((Vec::new(), Vec::new())),
            |a, b| {
                let (mut a_cells, mut a_per_class) = a?;
                let (b_cells, b_per_class) = b?;
                a_cells.extend(b_cells);
                a_per_class.extend(b_per_class);
                Ok((a_cells, a_per_class))
            },
        )
}

/// From a per-class-keyed score map, compute one `PerClassRow` per
/// `(dataset, config, peak_count, query_pathway)` and one micro-pooled
/// `CellRow` per `(dataset, config, peak_count)` by concatenating samples
/// across query pathways. Sample buffers are moved into the pooled map so
/// the function ends with one allocation per pooled cell rather than two.
fn compute_rows_from_groups(groups: PerClassGroups) -> (Vec<CellRow>, Vec<PerClassRow>) {
    let mut per_class_rows: Vec<PerClassRow> = Vec::with_capacity(groups.len());
    let mut pooled: HashMap<(String, String, u64), Vec<(f64, bool)>> = HashMap::new();
    for ((dataset, config, peak_count, pathway), mut samples) in groups {
        per_class_rows.push(per_class_row_from_samples(
            &dataset,
            &config,
            peak_count,
            &pathway,
            &mut samples,
        ));
        pooled
            .entry((dataset, config, peak_count))
            .or_default()
            .extend(samples);
    }
    let cell_rows: Vec<CellRow> = pooled
        .into_par_iter()
        .map(|((dataset, config, peak_count), mut samples)| {
            let n_pos = samples.iter().filter(|(_, label)| *label).count();
            let n_neg = samples.len() - n_pos;
            let auroc_value = auroc(&mut samples);
            let auprc_value = auprc(&mut samples);
            CellRow {
                dataset,
                config,
                peak_count,
                auroc: auroc_value,
                auprc: auprc_value,
                n_positives: n_pos as u64,
                n_negatives: n_neg as u64,
            }
        })
        .collect();
    (cell_rows, per_class_rows)
}

/// Reusable AUROC / AUPRC computation for one per-class score bucket.
fn per_class_row_from_samples(
    dataset: &str,
    config: &str,
    peak_count: u64,
    pathway: &str,
    samples: &mut [(f64, bool)],
) -> PerClassRow {
    let n_pos = samples.iter().filter(|(_, label)| *label).count();
    let n_neg = samples.len() - n_pos;
    let auroc_value = auroc(samples);
    let auprc_value = auprc(samples);
    PerClassRow {
        dataset: dataset.to_string(),
        config: config.to_string(),
        peak_count,
        pathway: pathway.to_string(),
        auroc: auroc_value,
        auprc: auprc_value,
        n_positives: n_pos as u64,
        n_negatives: n_neg as u64,
    }
}

fn observe_batch(batch: &RecordBatch, groups: &mut PerClassGroups) -> Result<()> {
    let datasets = required_column::<StringArray>(batch, "dataset")?;
    let configs = required_column::<StringArray>(batch, "config")?;
    let peak_counts = required_column::<UInt64Array>(batch, "peak_count")?;
    let query_pathways = required_column::<StringArray>(batch, "query_npc_pathway")?;
    let candidate_pathways = required_column::<StringArray>(batch, "candidate_npc_pathway")?;
    let scores = required_column::<Float64Array>(batch, "score")?;

    for row in 0..batch.num_rows() {
        // Skip rows with no query pathway label — we can't define a
        // same-class label without ground truth.
        if query_pathways.is_null(row) {
            continue;
        }
        let dataset = datasets.value(row).to_string();
        let config = configs.value(row).to_string();
        let peak_count = peak_counts.value(row);
        let query_pathway = query_pathways.value(row).to_string();
        let label = candidate_pathways.value(row) == query_pathway.as_str();
        let score = scores.value(row);
        if !score.is_finite() {
            continue;
        }
        groups
            .entry((dataset, config, peak_count, query_pathway))
            .or_default()
            .push((score, label));
    }
    Ok(())
}

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

fn summarize_per_config(cell_rows: &[CellRow]) -> Vec<SummaryRow> {
    let mut by_config: HashMap<(String, String), (f64, f64, u64)> = HashMap::new();
    for row in cell_rows {
        if !row.auroc.is_finite() || !row.auprc.is_finite() {
            continue;
        }
        let entry = by_config
            .entry((row.dataset.clone(), row.config.clone()))
            .or_default();
        entry.0 += row.auroc;
        entry.1 += row.auprc;
        entry.2 += 1;
    }
    let mut summary: Vec<SummaryRow> = by_config
        .into_iter()
        .map(|((dataset, config), (sum_auroc, sum_auprc, count))| {
            let count_f = count as f64;
            SummaryRow {
                dataset,
                config,
                mean_auroc: sum_auroc / count_f,
                mean_auprc: sum_auprc / count_f,
                n_peak_counts: count,
            }
        })
        .collect();
    summary.sort_by(|a, b| a.dataset.cmp(&b.dataset).then(a.config.cmp(&b.config)));
    summary
}

fn cell_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("dataset", DataType::Utf8, false),
        Field::new("config", DataType::Utf8, false),
        Field::new("peak_count", DataType::UInt64, false),
        Field::new("auroc", DataType::Float64, true),
        Field::new("auprc", DataType::Float64, true),
        Field::new("n_positives", DataType::UInt64, false),
        Field::new("n_negatives", DataType::UInt64, false),
    ]))
}

fn per_class_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("dataset", DataType::Utf8, false),
        Field::new("config", DataType::Utf8, false),
        Field::new("peak_count", DataType::UInt64, false),
        Field::new("pathway", DataType::Utf8, false),
        Field::new("auroc", DataType::Float64, true),
        Field::new("auprc", DataType::Float64, true),
        Field::new("n_positives", DataType::UInt64, false),
        Field::new("n_negatives", DataType::UInt64, false),
    ]))
}

fn summary_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("dataset", DataType::Utf8, false),
        Field::new("config", DataType::Utf8, false),
        Field::new("mean_auroc", DataType::Float64, false),
        Field::new("mean_auprc", DataType::Float64, false),
        Field::new("n_peak_counts", DataType::UInt64, false),
    ]))
}

fn write_cell_rows(output_dir: &Path, rows: &[CellRow]) -> Result<()> {
    let path = output_dir.join("pathway_discriminability.parquet");
    let datasets: StringArray = rows.iter().map(|r| Some(r.dataset.as_str())).collect();
    let configs: StringArray = rows.iter().map(|r| Some(r.config.as_str())).collect();
    let peak_counts: UInt64Array = rows.iter().map(|r| r.peak_count).collect();
    let auroc_values: Float64Array = rows
        .iter()
        .map(|r| {
            if r.auroc.is_finite() {
                Some(r.auroc)
            } else {
                None
            }
        })
        .collect();
    let auprc_values: Float64Array = rows
        .iter()
        .map(|r| {
            if r.auprc.is_finite() {
                Some(r.auprc)
            } else {
                None
            }
        })
        .collect();
    let n_pos: UInt64Array = rows.iter().map(|r| r.n_positives).collect();
    let n_neg: UInt64Array = rows.iter().map(|r| r.n_negatives).collect();
    let schema = cell_schema();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(datasets),
            Arc::new(configs),
            Arc::new(peak_counts),
            Arc::new(auroc_values),
            Arc::new(auprc_values),
            Arc::new(n_pos),
            Arc::new(n_neg),
        ],
    )
    .with_context(|| format!("building record batch for {}", path.display()))?;
    let file = fs::File::create(&path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(parquet_writer_props()))
        .with_context(|| format!("opening writer for {}", path.display()))?;
    writer
        .write(&batch)
        .with_context(|| format!("writing rows to {}", path.display()))?;
    writer
        .close()
        .with_context(|| format!("finalizing {}", path.display()))?;
    Ok(())
}

fn write_per_class_rows(output_dir: &Path, rows: &[PerClassRow]) -> Result<()> {
    let path = output_dir.join("pathway_discriminability_per_class.parquet");
    let datasets: StringArray = rows.iter().map(|r| Some(r.dataset.as_str())).collect();
    let configs: StringArray = rows.iter().map(|r| Some(r.config.as_str())).collect();
    let peak_counts: UInt64Array = rows.iter().map(|r| r.peak_count).collect();
    let pathways: StringArray = rows.iter().map(|r| Some(r.pathway.as_str())).collect();
    let auroc_values: Float64Array = rows
        .iter()
        .map(|r| r.auroc.is_finite().then_some(r.auroc))
        .collect();
    let auprc_values: Float64Array = rows
        .iter()
        .map(|r| r.auprc.is_finite().then_some(r.auprc))
        .collect();
    let n_pos: UInt64Array = rows.iter().map(|r| r.n_positives).collect();
    let n_neg: UInt64Array = rows.iter().map(|r| r.n_negatives).collect();
    let schema = per_class_schema();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(datasets),
            Arc::new(configs),
            Arc::new(peak_counts),
            Arc::new(pathways),
            Arc::new(auroc_values),
            Arc::new(auprc_values),
            Arc::new(n_pos),
            Arc::new(n_neg),
        ],
    )
    .with_context(|| format!("building record batch for {}", path.display()))?;
    let file = fs::File::create(&path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(parquet_writer_props()))
        .with_context(|| format!("opening writer for {}", path.display()))?;
    writer
        .write(&batch)
        .with_context(|| format!("writing rows to {}", path.display()))?;
    writer
        .close()
        .with_context(|| format!("finalizing {}", path.display()))?;
    Ok(())
}

fn write_summary_rows(output_dir: &Path, rows: &[SummaryRow]) -> Result<()> {
    let path = output_dir.join("pathway_discriminability_summary.parquet");
    if rows.is_empty() {
        bail!(
            "no per-config summary rows to write; check {}/pathway_scores.parquet",
            output_dir.display()
        );
    }
    let datasets: StringArray = rows.iter().map(|r| Some(r.dataset.as_str())).collect();
    let configs: StringArray = rows.iter().map(|r| Some(r.config.as_str())).collect();
    let mean_auroc: Float64Array = rows.iter().map(|r| Some(r.mean_auroc)).collect();
    let mean_auprc: Float64Array = rows.iter().map(|r| Some(r.mean_auprc)).collect();
    let n_peaks: UInt64Array = rows.iter().map(|r| r.n_peak_counts).collect();
    let schema = summary_schema();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(datasets),
            Arc::new(configs),
            Arc::new(mean_auroc),
            Arc::new(mean_auprc),
            Arc::new(n_peaks),
        ],
    )
    .with_context(|| format!("building record batch for {}", path.display()))?;
    let file = fs::File::create(&path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(parquet_writer_props()))
        .with_context(|| format!("opening writer for {}", path.display()))?;
    writer
        .write(&batch)
        .with_context(|| format!("writing rows to {}", path.display()))?;
    writer
        .close()
        .with_context(|| format!("finalizing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{auprc, auroc};

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn auroc_perfect_ranking_is_one() {
        let mut samples = vec![(0.1, false), (0.2, false), (0.8, true), (0.9, true)];
        assert!(approx(auroc(&mut samples), 1.0, 1.0e-12));
    }

    #[test]
    fn auroc_inverse_ranking_is_zero() {
        let mut samples = vec![(0.1, true), (0.2, true), (0.8, false), (0.9, false)];
        assert!(approx(auroc(&mut samples), 0.0, 1.0e-12));
    }

    #[test]
    fn auroc_balanced_positions_is_half() {
        // Positives at ranks 2 and 3 in a 4-sample sweep: sum_pos = 5,
        // AUROC = (5 - 2*3/2) / (2*2) = 0.5.
        let mut samples = vec![(0.10, false), (0.20, true), (0.30, true), (0.40, false)];
        assert!(approx(auroc(&mut samples), 0.5, 1.0e-12));
    }

    #[test]
    fn auroc_identical_scores_with_mixed_labels_is_half() {
        // All scores tied: every positive gets the same average rank ⇒ AUROC = 0.5.
        let mut samples = vec![(0.5, true), (0.5, false), (0.5, true), (0.5, false)];
        assert!(approx(auroc(&mut samples), 0.5, 1.0e-12));
    }

    #[test]
    fn auroc_hand_computed_with_ties() {
        // Scores [1, 1, 2, 3] with labels [T, F, T, F].
        // Ascending sort same as input order.
        // Ranks with ties: positions 1,2 tied → avg rank 1.5; pos 3 → 3; pos 4 → 4.
        // Positive ranks: 1.5 (idx 0) + 3 (idx 2) = 4.5.
        // n_pos = 2, n_neg = 2 → AUROC = (4.5 - 2*3/2) / (2*2) = (4.5 - 3)/4 = 0.375.
        let mut samples = vec![(1.0, true), (1.0, false), (2.0, true), (3.0, false)];
        let expected = 0.375;
        let got = auroc(&mut samples);
        assert!(
            approx(got, expected, 1.0e-12),
            "expected {expected}, got {got}"
        );
    }

    #[test]
    fn auroc_returns_nan_for_one_class_only() {
        let mut all_pos = vec![(0.1_f64, true), (0.2, true)];
        let mut all_neg = vec![(0.1_f64, false), (0.2, false)];
        assert!(auroc(&mut all_pos).is_nan());
        assert!(auroc(&mut all_neg).is_nan());
    }

    #[test]
    fn auprc_perfect_ranking_is_one() {
        let mut samples = vec![(0.9, true), (0.8, true), (0.2, false), (0.1, false)];
        assert!(approx(auprc(&mut samples), 1.0, 1.0e-12));
    }

    #[test]
    fn auprc_inverse_ranking_equals_one_over_n() {
        // With all positives ranked last, AP becomes the average of 1/(N-k+1)
        // for k = 1..n_pos over reversed positions — effectively very low.
        let mut samples = vec![(0.9, false), (0.8, false), (0.2, true), (0.1, true)];
        // Sorted desc: (0.9,F)(0.8,F)(0.2,T)(0.1,T)
        // At i=2 (1-indexed 3): tp=1, precision=1/3 ≈ 0.333, +1/2 * 0.333
        // At i=3 (1-indexed 4): tp=2, precision=2/4 = 0.5, +1/2 * 0.5
        // AP = 0.5*(1/3) + 0.5*(1/2) = 1/6 + 1/4 = 5/12 ≈ 0.4167
        let expected = 5.0 / 12.0;
        let got = auprc(&mut samples);
        assert!(
            approx(got, expected, 1.0e-12),
            "expected {expected}, got {got}"
        );
    }

    #[test]
    fn auprc_at_base_rate_for_random_ranking() {
        // For ranked perfectly alternating (one positive after each negative
        // starting from the top), AP picks up at each positive position.
        let mut samples = vec![
            (0.9, true),
            (0.8, false),
            (0.7, true),
            (0.6, false),
            (0.5, true),
            (0.4, false),
        ];
        // Hand: positives at positions 1, 3, 5.
        // P@1 = 1/1, P@3 = 2/3, P@5 = 3/5. AP = (1 + 2/3 + 3/5)/3 = (15 + 10 + 9)/45 = 34/45.
        let expected = 34.0 / 45.0;
        let got = auprc(&mut samples);
        assert!(
            approx(got, expected, 1.0e-12),
            "expected {expected}, got {got}"
        );
    }

    #[test]
    fn auprc_returns_nan_when_no_positives() {
        let mut samples = vec![(0.1_f64, false), (0.2, false), (0.3, false)];
        assert!(auprc(&mut samples).is_nan());
    }

    #[test]
    fn auprc_handles_ties_consistently() {
        // All same score with mixed labels: precision at the tie block is
        // tp_total / (tp+fp)_total. n_pos=2, total=4 in tie block →
        // precision = 2/4 = 0.5. AP = 0.5 * (2/2) = 0.5.
        let mut samples = vec![(0.5, true), (0.5, false), (0.5, true), (0.5, false)];
        let expected = 0.5;
        let got = auprc(&mut samples);
        assert!(
            approx(got, expected, 1.0e-12),
            "expected {expected}, got {got}"
        );
    }
}

//! Per-config "diversity" measured as the mean Kolmogorov–Smirnov statistic
//! across all off-diagonal cells of a config's 128 × 128 score-distribution
//! grid.
//!
//! Intuition: a config whose score distributions shift a lot as peak count
//! changes — and that produces large pairwise CDF gaps — has a higher mean
//! D and is more "diverse". A config whose distributions barely change has
//! a small mean D. Diagonal cells are skipped (D = 0 by self-comparison).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::missing_docs_in_private_items
)]

use std::{fs, path::Path, sync::Arc};

use anyhow::{Context, Result, bail};
use arrow_array::{Float64Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use ndarray::Array3;
use ndarray_npy::NpzReader;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::progress::ScanProgress;

/// Read `distribution_grid.npz` and `distribution_grid_configs.parquet`,
/// compute the per-config mean / stddev of D over all off-diagonal cells,
/// rank configs from most to least diverse, and write
/// `config_diversity.parquet` to `output_dir`. Also prints the ranked table
/// to stdout for quick inspection.
///
/// # Errors
///
/// Returns an error when either input artifact is missing or malformed, or
/// when the output parquet cannot be written.
pub fn write_config_diversity(output_dir: &Path, progress: &ScanProgress) -> Result<()> {
    let read_progress = progress.spinner("reading distribution_grid.npz");
    let (configs, ks_statistic) = read_inputs(output_dir)?;
    read_progress.finish();

    let n_configs = configs.len();
    if ks_statistic.shape()[0] != n_configs {
        bail!(
            "config axis mismatch: npz has {} configs, parquet has {}",
            ks_statistic.shape()[0],
            n_configs
        );
    }

    let mut rows: Vec<DiversityRow> = (0..n_configs)
        .map(|c| {
            let slice = ks_statistic.index_axis(ndarray::Axis(0), c);
            let (mean, stddev, n_cells) = off_diagonal_mean_and_stddev(&slice);
            let d10_peak = crate::visualize::ks_statistic_asymptote(&slice, 0.10);
            let d05_peak = crate::visualize::ks_statistic_asymptote(&slice, 0.05);
            let d01_peak = crate::visualize::ks_statistic_asymptote(&slice, 0.01);
            DiversityRow {
                config: configs[c].clone(),
                mean_d: mean,
                stddev_d: stddev,
                n_cells,
                d10_peak,
                d05_peak,
                d01_peak,
            }
        })
        .collect();
    rows.sort_by(|a, b| {
        b.mean_d
            .partial_cmp(&a.mean_d)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    print_ranking(&rows);

    let write_progress = progress.spinner("writing config_diversity.parquet");
    write_diversity_parquet(output_dir, &rows)?;
    write_progress.finish();
    Ok(())
}

struct DiversityRow {
    config: String,
    mean_d: f64,
    stddev_d: f64,
    n_cells: u64,
    /// Peak count at which the `D = 0.10` contour reaches its right-edge
    /// asymptote (`None` if the contour never separates from the diagonal).
    d10_peak: Option<i32>,
    /// Same, for `D = 0.05`.
    d05_peak: Option<i32>,
    /// Same, for `D = 0.01`.
    d01_peak: Option<i32>,
}

fn off_diagonal_mean_and_stddev(grid: &ndarray::ArrayView2<f64>) -> (f64, f64, u64) {
    let (n_rows, n_cols) = grid.dim();
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut count: u64 = 0;
    for r in 0..n_rows {
        for c in 0..n_cols {
            if r == c {
                continue;
            }
            let value = grid[(r, c)];
            if !value.is_finite() {
                continue;
            }
            sum += value;
            sum_sq = value.mul_add(value, sum_sq);
            count += 1;
        }
    }
    if count == 0 {
        return (f64::NAN, f64::NAN, 0);
    }
    let mean = sum / count as f64;
    let variance = mean.mul_add(-mean, sum_sq / count as f64);
    (mean, variance.max(0.0).sqrt(), count)
}

fn read_inputs(output_dir: &Path) -> Result<(Vec<String>, Array3<f64>)> {
    let npz_path = output_dir.join("distribution_grid.npz");
    let file =
        fs::File::open(&npz_path).with_context(|| format!("opening {}", npz_path.display()))?;
    let mut reader =
        NpzReader::new(file).with_context(|| format!("reading {}", npz_path.display()))?;
    let ks_statistic: Array3<f64> = reader.by_name("ks_statistic.npy")?;

    let configs_path = output_dir.join("distribution_grid_configs.parquet");
    let configs = read_config_labels(&configs_path)?;
    Ok((configs, ks_statistic))
}

fn read_config_labels(path: &Path) -> Result<Vec<String>> {
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading metadata from {}", path.display()))?
        .build()
        .with_context(|| format!("building reader for {}", path.display()))?;
    let mut pairs: Vec<(usize, String)> = Vec::new();
    for batch in reader {
        let batch = batch?;
        let indices = batch
            .column_by_name("config_index")
            .context("missing config_index column")?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .context("config_index column has unexpected type")?;
        let names = batch
            .column_by_name("config")
            .context("missing config column")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("config column has unexpected type")?;
        for row in 0..batch.num_rows() {
            let index =
                usize::try_from(indices.value(row)).context("config_index does not fit usize")?;
            pairs.push((index, names.value(row).to_string()));
        }
    }
    pairs.sort_by_key(|(i, _)| *i);
    Ok(pairs.into_iter().map(|(_, name)| name).collect())
}

fn print_ranking(rows: &[DiversityRow]) {
    println!("\n== per-config diversity (mean off-diagonal KS statistic) ==");
    println!(
        "{:>4}  {:<50}  {:>10}  {:>10}  {:>9}  {:>9}  {:>9}",
        "rank", "config", "mean D", "stddev D", "D=0.10 pk", "D=0.05 pk", "D=0.01 pk"
    );
    let fmt_peak = |peak: Option<i32>| peak.map_or_else(|| "—".to_string(), |v| v.to_string());
    for (rank, row) in rows.iter().enumerate() {
        println!(
            "{:>4}  {:<50}  {:>10.5}  {:>10.5}  {:>9}  {:>9}  {:>9}",
            rank + 1,
            row.config,
            row.mean_d,
            row.stddev_d,
            fmt_peak(row.d10_peak),
            fmt_peak(row.d05_peak),
            fmt_peak(row.d01_peak),
        );
    }
    println!();
}

fn diversity_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("rank", DataType::UInt64, false),
        Field::new("config", DataType::Utf8, false),
        Field::new("mean_d", DataType::Float64, false),
        Field::new("stddev_d", DataType::Float64, false),
        Field::new("n_cells", DataType::UInt64, false),
        Field::new("d10_peak", DataType::Int64, true),
        Field::new("d05_peak", DataType::Int64, true),
        Field::new("d01_peak", DataType::Int64, true),
    ]))
}

fn write_diversity_parquet(output_dir: &Path, rows: &[DiversityRow]) -> Result<()> {
    let path = output_dir.join("config_diversity.parquet");
    let ranks: UInt64Array = (1..=rows.len() as u64).collect();
    let configs: StringArray = rows.iter().map(|r| Some(r.config.as_str())).collect();
    let mean_d: Float64Array = rows.iter().map(|r| Some(r.mean_d)).collect();
    let stddev_d: Float64Array = rows.iter().map(|r| Some(r.stddev_d)).collect();
    let n_cells: UInt64Array = rows.iter().map(|r| r.n_cells).collect();
    let d10_peak: arrow_array::Int64Array =
        rows.iter().map(|r| r.d10_peak.map(i64::from)).collect();
    let d05_peak: arrow_array::Int64Array =
        rows.iter().map(|r| r.d05_peak.map(i64::from)).collect();
    let d01_peak: arrow_array::Int64Array =
        rows.iter().map(|r| r.d01_peak.map(i64::from)).collect();
    let schema = diversity_schema();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(ranks),
            Arc::new(configs),
            Arc::new(mean_d),
            Arc::new(stddev_d),
            Arc::new(n_cells),
            Arc::new(d10_peak),
            Arc::new(d05_peak),
            Arc::new(d01_peak),
        ],
    )?;
    let file = fs::File::create(&path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, None)
        .with_context(|| format!("opening writer for {}", path.display()))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::off_diagonal_mean_and_stddev;
    use ndarray::array;

    #[test]
    fn off_diagonal_mean_skips_diagonal() {
        // 3x3: diagonal 0s, off-diagonal all 0.5 → mean = 0.5, stddev = 0.
        let grid = array![[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]];
        let (mean, stddev, n) = off_diagonal_mean_and_stddev(&grid.view());
        assert!((mean - 0.5).abs() < 1.0e-12);
        assert!(stddev.abs() < 1.0e-12);
        assert_eq!(n, 6);
    }

    #[test]
    fn off_diagonal_mean_handles_nan_cells() {
        let grid = array![[0.0, 0.1, f64::NAN], [0.1, 0.0, 0.3], [f64::NAN, 0.3, 0.0]];
        let (mean, _, n) = off_diagonal_mean_and_stddev(&grid.view());
        // valid off-diagonal: [0.1, 0.1, 0.3, 0.3] → mean = 0.2
        assert!((mean - 0.2).abs() < 1.0e-12);
        assert_eq!(n, 4);
    }
}

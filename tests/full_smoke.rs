//! End-to-end smoke test for the command-line scan workflow.

use std::{
    error::Error,
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use clap::{Parser, error::ErrorKind};
use ndarray::{Array1, Array3};
use ndarray_npy::NpzReader;
use parquet::file::reader::{FileReader, SerializedFileReader};
use spectral_similarities_by_peaks::{cli::Cli, run};

#[test]
/// The synthetic scan writes all expected binary artifacts.
fn full_scan_smoke_test_produces_expected_artifacts() -> Result<(), Box<dyn Error>> {
    let root = smoke_root()?;
    let data_dir = root.join("data");
    let output_dir = root.join("out");
    fs::create_dir_all(&data_dir)?;

    let cli = Cli::try_parse_from([
        "spectral-similarities-by-peaks",
        "scan",
        "--dataset",
        "synthetic-smoke",
        "--data-dir",
        data_dir
            .to_str()
            .ok_or("temporary data directory path is not valid UTF-8")?,
        "--output-dir",
        output_dir
            .to_str()
            .ok_or("temporary output directory path is not valid UTF-8")?,
        "--similarity-config",
        "cosine:0.0:1.0",
        "--similarity-config",
        "cosine:1.0:0.5",
        "--similarity-config",
        "entropy:0.0:1.0:true",
        "--neighbors",
        "3",
        "--mz-tolerance",
        "0.05",
        "--histogram-bins",
        "5",
        "--pathway-representatives-per-class",
        "2",
        "--row-sample-size",
        "12",
        "--reference-sample-size",
        "18",
        "--seed",
        "42",
    ])?;
    run::run(cli)?;

    assert_parquet_rows(&output_dir.join("similarities.parquet"), 13_824)?;
    assert_parquet_rows(&output_dir.join("distribution_summary.parquet"), 384)?;
    assert_parquet_rows(&output_dir.join("distribution_histograms.parquet"), 1_920)?;
    assert_parquet_rows(&output_dir.join("distribution_tests.parquet"), 381)?;
    assert_parquet_rows(&output_dir.join("distribution_grid.parquet"), 49_152)?;
    assert_parquet_rows(&output_dir.join("distribution_grid_configs.parquet"), 3)?;
    assert_parquet_rows(&output_dir.join("pathway_scores.parquet"), 12_288)?;
    assert_parquet_rows(&output_dir.join("pathway_predictions.parquet"), 3_072)?;
    assert_grid_npz_shapes(&output_dir.join("distribution_grid.npz"))?;
    assert_heatmap_artifacts(&output_dir)?;

    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
/// The top-level command help is generated successfully.
fn top_level_help_is_available() -> Result<(), Box<dyn Error>> {
    let Err(error) = Cli::try_parse_from(["spectral-similarities-by-peaks", "--help"]) else {
        return Err(std::io::Error::other("help should short-circuit parsing").into());
    };
    assert_eq!(error.kind(), ErrorKind::DisplayHelp);
    let stdout = error.to_string();
    assert!(
        stdout.contains("Commands:"),
        "unexpected help output: {stdout}"
    );
    assert!(stdout.contains("scan"), "missing scan command: {stdout}");
    Ok(())
}

#[test]
/// The scan subcommand help is generated successfully.
fn scan_help_is_available() -> Result<(), Box<dyn Error>> {
    let Err(error) = Cli::try_parse_from(["spectral-similarities-by-peaks", "scan", "--help"])
    else {
        return Err(std::io::Error::other("scan help should short-circuit parsing").into());
    };
    assert_eq!(error.kind(), ErrorKind::DisplayHelp);
    let stdout = error.to_string();
    assert!(
        stdout.contains("--dataset"),
        "missing dataset flag: {stdout}"
    );
    assert!(
        stdout.contains("--reference-sample-size"),
        "missing reference sampling flag: {stdout}"
    );
    assert!(
        !stdout.contains("--peak-counts"),
        "peak-count narrowing must not be exposed: {stdout}"
    );
    Ok(())
}

/// Return a unique temporary root for one smoke-test invocation.
fn smoke_root() -> Result<PathBuf, Box<dyn Error>> {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?;
    Ok(std::env::temp_dir().join(format!(
        "spectral-similarities-by-peaks-smoke-{}-{}",
        std::process::id(),
        timestamp.as_nanos()
    )))
}

/// Assert that a Parquet file exists, is non-empty, and has the expected row count.
fn assert_parquet_rows(path: &Path, expected_rows: usize) -> Result<(), Box<dyn Error>> {
    let metadata = fs::metadata(path)?;
    assert!(metadata.len() > 0, "{} is empty", path.display());

    let file = fs::File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let rows = usize::try_from(reader.metadata().file_metadata().num_rows())?;
    assert_eq!(
        rows,
        expected_rows,
        "{} has an unexpected number of rows",
        path.display()
    );
    Ok(())
}

/// Assert that the dense full-grid `NumPy` artifact has the expected axes.
fn assert_grid_npz_shapes(path: &Path) -> Result<(), Box<dyn Error>> {
    let file = fs::File::open(path)?;
    let mut reader = NpzReader::new(file)?;
    let peak_counts: Array1<u64> = reader.by_name("peak_counts.npy")?;
    let ks_statistic: Array3<f64> = reader.by_name("ks_statistic.npy")?;
    let ks_pvalue: Array3<f64> = reader.by_name("ks_pvalue_asymptotic.npy")?;
    let wasserstein: Array3<f64> = reader.by_name("wasserstein_1d.npy")?;
    let mean_delta: Array3<f64> = reader.by_name("mean_delta.npy")?;

    assert_eq!(peak_counts.len(), 128);
    assert_eq!(ks_statistic.shape(), &[3, 128, 128]);
    assert_eq!(ks_pvalue.shape(), &[3, 128, 128]);
    assert_eq!(wasserstein.shape(), &[3, 128, 128]);
    assert_eq!(mean_delta.shape(), &[3, 128, 128]);
    Ok(())
}

/// Assert that every smoke-test heatmap was written in SVG and PNG form.
fn assert_heatmap_artifacts(output_dir: &Path) -> Result<(), Box<dyn Error>> {
    for config in [
        "cosine_mz0.000_int1.000",
        "cosine_mz1.000_int0.500",
        "entropy_mz0.000_int1.000_weightedtrue",
    ] {
        for metric in [
            "mean_delta",
            "ks_statistic",
            "ks_pvalue_asymptotic",
            "wasserstein_1d",
        ] {
            let stem = output_dir.join("heatmaps").join(config).join(metric);
            assert_svg_artifact(&stem.with_extension("svg"))?;
            assert_png_artifact(&stem.with_extension("png"))?;
        }
    }
    Ok(())
}

/// Assert that an SVG heatmap exists and contains an SVG root.
fn assert_svg_artifact(path: &Path) -> Result<(), Box<dyn Error>> {
    let metadata = fs::metadata(path)?;
    assert!(metadata.len() > 0, "{} is empty", path.display());
    let content = fs::read_to_string(path)?;
    assert!(content.contains("<svg"), "{} is not SVG", path.display());
    Ok(())
}

/// Assert that a PNG heatmap exists and starts with the PNG signature.
fn assert_png_artifact(path: &Path) -> Result<(), Box<dyn Error>> {
    let metadata = fs::metadata(path)?;
    assert!(metadata.len() > 0, "{} is empty", path.display());
    let mut signature = [0_u8; 8];
    fs::File::open(path)?.read_exact(&mut signature)?;
    assert_eq!(
        signature,
        [137, 80, 78, 71, 13, 10, 26, 10],
        "{} is not PNG",
        path.display()
    );
    Ok(())
}

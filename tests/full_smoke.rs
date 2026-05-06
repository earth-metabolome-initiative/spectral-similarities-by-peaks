//! End-to-end smoke test for the command-line scan workflow.

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use clap::{Parser, error::ErrorKind};
use spectral_similarities_by_peaks::{cli::Cli, run};

#[test]
/// The synthetic scan writes all expected CSV artifacts.
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
        "--peak-counts",
        "4,8,16",
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

    assert_csv_rows(&output_dir.join("similarities.csv"), 324)?;
    assert_csv_rows(&output_dir.join("distribution_summary.csv"), 9)?;
    assert_csv_rows(&output_dir.join("distribution_histograms.csv"), 45)?;
    assert_csv_rows(&output_dir.join("distribution_tests.csv"), 6)?;
    assert_csv_rows(&output_dir.join("distribution_grid.csv"), 27)?;
    assert_csv_rows(&output_dir.join("pathway_scores.csv"), 288)?;
    assert_csv_rows(&output_dir.join("pathway_predictions.csv"), 72)?;

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

/// Assert that a CSV file exists, is non-empty, and has the expected row count.
fn assert_csv_rows(path: &Path, expected_rows: usize) -> Result<(), Box<dyn Error>> {
    let metadata = fs::metadata(path)?;
    assert!(metadata.len() > 0, "{} is empty", path.display());

    let mut reader = csv::Reader::from_path(path)?;
    let rows = reader.records().collect::<Result<Vec<_>, _>>()?;
    assert_eq!(
        rows.len(),
        expected_rows,
        "{} has an unexpected number of rows",
        path.display()
    );
    Ok(())
}

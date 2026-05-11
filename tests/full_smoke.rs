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

    let cli = smoke_scan_cli(&data_dir, &output_dir)?;
    run::run(cli)?;
    assert_distribution_checkpoints(&output_dir)?;

    let cli = smoke_scan_cli(&data_dir, &output_dir)?;
    run::run(cli)?;

    assert!(
        !output_dir.join("similarities.parquet").exists(),
        "raw similarity hits should not be persisted"
    );
    assert_parquet_rows(&output_dir.join("distribution_summary.parquet"), 2_304)?;
    assert_parquet_rows(&output_dir.join("distribution_histograms.parquet"), 11_520)?;
    assert_parquet_rows(&output_dir.join("distribution_tests.parquet"), 2_286)?;
    assert_parquet_rows(&output_dir.join("distribution_grid.parquet"), 294_912)?;
    assert_parquet_rows(&output_dir.join("distribution_grid_configs.parquet"), 18)?;
    assert_parquet_rows(&output_dir.join("pathway_scores.parquet"), 110_592)?;
    assert_parquet_rows(&output_dir.join("pathway_predictions.parquet"), 27_648)?;
    assert_parquet_rows(
        &output_dir.join("pathway_prediction_metrics.parquet"),
        11_520,
    )?;
    assert_parquet_rows(
        &output_dir.join("pathway_prediction_distribution_grid.parquet"),
        294_912,
    )?;
    assert_parquet_rows(
        &output_dir.join("pathway_prediction_distribution_grid_configs.parquet"),
        18,
    )?;
    assert_grid_npz_shapes(&output_dir.join("distribution_grid.npz"))?;
    assert_pathway_grid_npz_shapes(&output_dir.join("pathway_prediction_distribution_grid.npz"))?;
    assert_heatmap_artifacts(&output_dir)?;
    assert_pathway_prediction_artifacts(&output_dir)?;

    fs::remove_dir_all(root)?;
    Ok(())
}

/// Build the deterministic synthetic scan command used by the full smoke test.
fn smoke_scan_cli(data_dir: &Path, output_dir: &Path) -> Result<Cli, Box<dyn Error>> {
    Ok(Cli::try_parse_from([
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
        "modified-cosine:0.0:1.0",
        "--similarity-config",
        "cosine:1.0:1.0",
        "--similarity-config",
        "modified-cosine:1.0:1.0",
        "--similarity-config",
        "cosine:0.0:0.5",
        "--similarity-config",
        "modified-cosine:0.0:0.5",
        "--similarity-config",
        "cosine:1.0:0.5",
        "--similarity-config",
        "modified-cosine:1.0:0.5",
        "--similarity-config",
        "cosine:0.0:0.25",
        "--similarity-config",
        "modified-cosine:0.0:0.25",
        "--similarity-config",
        "cosine:1.0:0.25",
        "--similarity-config",
        "modified-cosine:1.0:0.25",
        "--similarity-config",
        "cosine:3.0:0.6",
        "--similarity-config",
        "modified-cosine:3.0:0.6",
        "--similarity-config",
        "entropy:0.0:1.0:true",
        "--similarity-config",
        "modified-entropy:0.0:1.0:true",
        "--similarity-config",
        "entropy:0.0:1.0:false",
        "--similarity-config",
        "modified-entropy:0.0:1.0:false",
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
    ])?)
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
    assert!(
        stdout.contains("render-pathway-artifacts"),
        "missing pathway artifact command: {stdout}"
    );
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

#[test]
/// Pathway artifact rendering reports missing prediction inputs.
fn render_pathway_artifacts_requires_predictions() -> Result<(), Box<dyn Error>> {
    let root = smoke_root()?;
    let output_dir = root.join("out");
    fs::create_dir_all(&output_dir)?;

    let cli = Cli::try_parse_from([
        "spectral-similarities-by-peaks",
        "render-pathway-artifacts",
        "--output-dir",
        output_dir
            .to_str()
            .ok_or("temporary output directory path is not valid UTF-8")?,
    ])?;
    let Err(error) = run::run(cli) else {
        return Err(std::io::Error::other("missing pathway predictions should fail").into());
    };
    let message = error.to_string();
    assert!(
        message.contains("pathway_predictions.parquet"),
        "unexpected error: {message}"
    );

    fs::remove_dir_all(root)?;
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

/// Assert that every score-distribution checkpoint was written.
fn assert_distribution_checkpoints(output_dir: &Path) -> Result<(), Box<dyn Error>> {
    for config in [
        "cosine_mz0.000_int1.000",
        "modified_cosine_mz0.000_int1.000",
        "cosine_mz1.000_int1.000",
        "modified_cosine_mz1.000_int1.000",
        "cosine_mz0.000_int0.500",
        "modified_cosine_mz0.000_int0.500",
        "cosine_mz1.000_int0.500",
        "modified_cosine_mz1.000_int0.500",
        "cosine_mz0.000_int0.250",
        "modified_cosine_mz0.000_int0.250",
        "cosine_mz1.000_int0.250",
        "modified_cosine_mz1.000_int0.250",
        "cosine_mz3.000_int0.600",
        "modified_cosine_mz3.000_int0.600",
        "entropy_mz0.000_int1.000_weightedtrue",
        "modified_entropy_mz0.000_int1.000_weightedtrue",
        "entropy_mz0.000_int1.000_weightedfalse",
        "modified_entropy_mz0.000_int1.000_weightedfalse",
    ] {
        for peak_count in 1..=128 {
            let path = output_dir
                .join("distributions")
                .join(config)
                .join(format!("top_{peak_count:03}.bincode.zst"));
            let metadata = fs::metadata(&path)?;
            assert!(metadata.len() > 0, "{} is empty", path.display());
        }
    }
    Ok(())
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
    assert_eq!(ks_statistic.shape(), &[18, 128, 128]);
    assert_eq!(ks_pvalue.shape(), &[18, 128, 128]);
    assert_eq!(wasserstein.shape(), &[18, 128, 128]);
    assert_eq!(mean_delta.shape(), &[18, 128, 128]);
    Ok(())
}

/// Assert that the pathway prediction dense-grid artifact has the expected axes.
fn assert_pathway_grid_npz_shapes(path: &Path) -> Result<(), Box<dyn Error>> {
    let file = fs::File::open(path)?;
    let mut reader = NpzReader::new(file)?;
    let peak_counts: Array1<u64> = reader.by_name("peak_counts.npy")?;
    let total_variation: Array3<f64> = reader.by_name("total_variation.npy")?;
    let jensen_shannon: Array3<f64> = reader.by_name("jensen_shannon_distance.npy")?;
    let hellinger: Array3<f64> = reader.by_name("hellinger_distance.npy")?;

    assert_eq!(peak_counts.len(), 128);
    assert_eq!(total_variation.shape(), &[18, 128, 128]);
    assert_eq!(jensen_shannon.shape(), &[18, 128, 128]);
    assert_eq!(hellinger.shape(), &[18, 128, 128]);
    Ok(())
}

/// Assert that every smoke-test heatmap was written in SVG and PNG form.
fn assert_heatmap_artifacts(output_dir: &Path) -> Result<(), Box<dyn Error>> {
    for config in [
        "cosine_mz0.000_int1.000",
        "modified_cosine_mz0.000_int1.000",
        "cosine_mz1.000_int1.000",
        "modified_cosine_mz1.000_int1.000",
        "cosine_mz0.000_int0.500",
        "modified_cosine_mz0.000_int0.500",
        "cosine_mz1.000_int0.500",
        "modified_cosine_mz1.000_int0.500",
        "cosine_mz0.000_int0.250",
        "modified_cosine_mz0.000_int0.250",
        "cosine_mz1.000_int0.250",
        "modified_cosine_mz1.000_int0.250",
        "cosine_mz3.000_int0.600",
        "modified_cosine_mz3.000_int0.600",
        "entropy_mz0.000_int1.000_weightedtrue",
        "modified_entropy_mz0.000_int1.000_weightedtrue",
        "entropy_mz0.000_int1.000_weightedfalse",
        "modified_entropy_mz0.000_int1.000_weightedfalse",
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

/// Assert that every smoke-test pathway prediction plot was written.
fn assert_pathway_prediction_artifacts(output_dir: &Path) -> Result<(), Box<dyn Error>> {
    for config in [
        "cosine_mz0.000_int1.000",
        "modified_cosine_mz0.000_int1.000",
        "cosine_mz1.000_int1.000",
        "modified_cosine_mz1.000_int1.000",
        "cosine_mz0.000_int0.500",
        "modified_cosine_mz0.000_int0.500",
        "cosine_mz1.000_int0.500",
        "modified_cosine_mz1.000_int0.500",
        "cosine_mz0.000_int0.250",
        "modified_cosine_mz0.000_int0.250",
        "cosine_mz1.000_int0.250",
        "modified_cosine_mz1.000_int0.250",
        "cosine_mz3.000_int0.600",
        "modified_cosine_mz3.000_int0.600",
        "entropy_mz0.000_int1.000_weightedtrue",
        "modified_entropy_mz0.000_int1.000_weightedtrue",
        "entropy_mz0.000_int1.000_weightedfalse",
        "modified_entropy_mz0.000_int1.000_weightedfalse",
    ] {
        for metric in [
            "total_variation",
            "jensen_shannon_distance",
            "hellinger_distance",
        ] {
            let stem = output_dir
                .join("pathway_prediction_heatmaps")
                .join(config)
                .join(metric);
            assert_svg_artifact(&stem.with_extension("svg"))?;
            assert_png_artifact(&stem.with_extension("png"))?;
        }
        for metric in ["accuracy", "mcc"] {
            let stem = output_dir
                .join("pathway_prediction_plots")
                .join(config)
                .join(metric);
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

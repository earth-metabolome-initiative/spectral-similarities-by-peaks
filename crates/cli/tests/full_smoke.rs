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

    let cli = smoke_compute_pathway_discriminability_cli(&output_dir)?;
    run::run(cli)?;
    assert_parquet_nonempty(&output_dir.join("pathway_discriminability.parquet"))?;
    assert_parquet_nonempty(&output_dir.join("pathway_discriminability_summary.parquet"))?;

    let cli = smoke_render_pathway_discriminability_cli(&output_dir)?;
    run::run(cli)?;
    assert_pathway_discriminability_artifacts(&output_dir)?;

    let cli = smoke_export_pathway_discriminability_json_cli(&output_dir)?;
    run::run(cli)?;
    assert_pathway_discriminability_json(&output_dir)?;

    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
/// Shard scans can be finalized into the same artifact family as a local scan.
fn shard_scan_finalize_smoke_test_produces_expected_artifacts() -> Result<(), Box<dyn Error>> {
    let root = smoke_root()?;
    let data_dir = root.join("data");
    let output_dir = root.join("out");
    fs::create_dir_all(&data_dir)?;

    for shard_index in 0..128 {
        let cli = smoke_single_config_shard_cli(&data_dir, &output_dir, shard_index)?;
        run::run(cli)?;
    }

    let cli = smoke_single_config_finalize_cli(&data_dir, &output_dir)?;
    run::run(cli)?;

    assert_parquet_rows(&output_dir.join("distribution_summary.parquet"), 128)?;
    assert_parquet_rows(&output_dir.join("distribution_histograms.parquet"), 384)?;
    assert_parquet_rows(&output_dir.join("distribution_tests.parquet"), 127)?;
    assert_parquet_rows(&output_dir.join("distribution_grid.parquet"), 16_384)?;
    assert_parquet_rows(&output_dir.join("distribution_grid_configs.parquet"), 1)?;
    assert_parquet_rows(&output_dir.join("pathway_scores.parquet"), 2_048)?;
    assert_parquet_rows(&output_dir.join("pathway_predictions.parquet"), 512)?;
    assert_parquet_nonempty(&output_dir.join("pathway_prediction_metrics.parquet"))?;
    assert_parquet_rows(
        &output_dir.join("pathway_prediction_distribution_grid.parquet"),
        16_384,
    )?;
    assert_parquet_rows(
        &output_dir.join("pathway_prediction_distribution_grid_configs.parquet"),
        1,
    )?;
    assert_grid_npz_shapes_with_configs(&output_dir.join("distribution_grid.npz"), 1)?;
    assert_pathway_grid_npz_shapes_with_configs(
        &output_dir.join("pathway_prediction_distribution_grid.npz"),
        1,
    )?;
    assert_single_config_heatmap_artifacts(&output_dir, "cosine_mz0.000_int1.000")?;
    assert_single_config_pathway_prediction_artifacts(&output_dir, "cosine_mz0.000_int1.000")?;

    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
/// Sharded finalize (one shard per config) followed by a merge produces the
/// same artifact family and row counts as the single-process finalize-scan.
fn sharded_finalize_smoke_test_matches_single_process() -> Result<(), Box<dyn Error>> {
    let root = smoke_root()?;
    let data_dir = root.join("data");
    let output_dir = root.join("out");
    fs::create_dir_all(&data_dir)?;

    for shard_index in 0..128 {
        let cli = smoke_single_config_shard_cli(&data_dir, &output_dir, shard_index)?;
        run::run(cli)?;
    }

    let cli = smoke_single_config_finalize_shard_cli(&data_dir, &output_dir, 0)?;
    run::run(cli)?;

    let cli = smoke_single_config_finalize_merge_cli(&data_dir, &output_dir)?;
    run::run(cli)?;

    assert_parquet_rows(&output_dir.join("distribution_summary.parquet"), 128)?;
    assert_parquet_rows(&output_dir.join("distribution_histograms.parquet"), 384)?;
    assert_parquet_rows(&output_dir.join("distribution_tests.parquet"), 127)?;
    assert_parquet_rows(&output_dir.join("distribution_grid.parquet"), 16_384)?;
    assert_parquet_rows(&output_dir.join("distribution_grid_configs.parquet"), 1)?;
    assert_parquet_rows(&output_dir.join("pathway_scores.parquet"), 2_048)?;
    assert_parquet_rows(&output_dir.join("pathway_predictions.parquet"), 512)?;
    assert_parquet_nonempty(&output_dir.join("pathway_prediction_metrics.parquet"))?;
    assert_parquet_rows(
        &output_dir.join("pathway_prediction_distribution_grid.parquet"),
        16_384,
    )?;
    assert_parquet_rows(
        &output_dir.join("pathway_prediction_distribution_grid_configs.parquet"),
        1,
    )?;
    assert_grid_npz_shapes_with_configs(&output_dir.join("distribution_grid.npz"), 1)?;
    assert_pathway_grid_npz_shapes_with_configs(
        &output_dir.join("pathway_prediction_distribution_grid.npz"),
        1,
    )?;
    assert_single_config_heatmap_artifacts(&output_dir, "cosine_mz0.000_int1.000")?;
    assert_single_config_pathway_prediction_artifacts(&output_dir, "cosine_mz0.000_int1.000")?;

    assert!(
        !output_dir.join("_finalize_shards").exists(),
        "finalize-merge should remove _finalize_shards/ on success"
    );

    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
/// When the heatmap renderer cannot find a font the finalize step must fail,
/// but the per-config Parquet artifacts and dense matrix outputs must still
/// land on disk fully closed and openable. Spawned in a subprocess so the
/// plotters font cache (a process-wide `OnceLock`) stays isolated.
fn distribution_parquets_survive_heatmap_font_failure() -> Result<(), Box<dyn Error>> {
    let root = smoke_root()?;
    let data_dir = root.join("data");
    let output_dir = root.join("out");
    fs::create_dir_all(&data_dir)?;

    for shard_index in 0..128 {
        let cli = smoke_single_config_shard_cli(&data_dir, &output_dir, shard_index)?;
        run::run(cli)?;
    }

    let data_dir_str = data_dir
        .to_str()
        .ok_or("temporary data directory path is not valid UTF-8")?;
    let output_dir_str = output_dir
        .to_str()
        .ok_or("temporary output directory path is not valid UTF-8")?;
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_spectral-similarities-by-peaks"))
        .env(
            "SPECTRAL_SIMILARITIES_FONT",
            "/this/path/does/not/exist.ttf",
        )
        .args([
            "finalize-scan",
            "--dataset",
            "synthetic-smoke",
            "--data-dir",
            data_dir_str,
            "--output-dir",
            output_dir_str,
            "--similarity-config",
            "cosine:0.0:1.0",
            "--neighbors",
            "2",
            "--mz-tolerance",
            "0.05",
            "--histogram-bins",
            "3",
            "--pathway-representatives-per-class",
            "1",
            "--row-sample-size",
            "4",
            "--reference-sample-size",
            "6",
            "--max-spectra",
            "8",
            "--seed",
            "42",
        ])
        .status()?;
    assert!(
        !status.success(),
        "finalize must fail when the heatmap font is missing"
    );

    assert_parquet_rows(&output_dir.join("distribution_summary.parquet"), 128)?;
    assert_parquet_rows(&output_dir.join("distribution_histograms.parquet"), 384)?;
    assert_parquet_rows(&output_dir.join("distribution_tests.parquet"), 127)?;
    assert_parquet_rows(&output_dir.join("distribution_grid.parquet"), 16_384)?;
    assert_parquet_rows(&output_dir.join("distribution_grid_configs.parquet"), 1)?;
    assert_parquet_rows(&output_dir.join("pathway_scores.parquet"), 2_048)?;
    assert_parquet_rows(&output_dir.join("pathway_predictions.parquet"), 512)?;
    assert!(
        output_dir.join("distribution_grid.npz").is_file(),
        "distribution_grid.npz must be written before heatmaps run"
    );

    let heatmaps_dir = output_dir.join("heatmaps");
    let heatmap_files = if heatmaps_dir.is_dir() {
        fs::read_dir(&heatmaps_dir).map_or(0, std::iter::Iterator::count)
    } else {
        0
    };
    assert_eq!(
        heatmap_files, 0,
        "heatmap renderer should not have produced any files"
    );
    assert!(
        !output_dir
            .join("pathway_prediction_metrics.parquet")
            .exists(),
        "pathway_prediction artifacts run after finalize and must not appear when finalize fails"
    );

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

/// Build the synthetic compute-pathway-discriminability command used by the smoke test.
fn smoke_compute_pathway_discriminability_cli(output_dir: &Path) -> Result<Cli, Box<dyn Error>> {
    Ok(Cli::try_parse_from([
        "spectral-similarities-by-peaks",
        "compute-pathway-discriminability",
        "--output-dir",
        output_dir
            .to_str()
            .ok_or("temporary output directory path is not valid UTF-8")?,
    ])?)
}

/// Build the synthetic render-pathway-discriminability command used by the smoke test.
fn smoke_render_pathway_discriminability_cli(output_dir: &Path) -> Result<Cli, Box<dyn Error>> {
    Ok(Cli::try_parse_from([
        "spectral-similarities-by-peaks",
        "render-pathway-discriminability",
        "--output-dir",
        output_dir
            .to_str()
            .ok_or("temporary output directory path is not valid UTF-8")?,
    ])?)
}

/// Build the synthetic export-pathway-discriminability-json command used by the smoke test.
fn smoke_export_pathway_discriminability_json_cli(
    output_dir: &Path,
) -> Result<Cli, Box<dyn Error>> {
    Ok(Cli::try_parse_from([
        "spectral-similarities-by-peaks",
        "export-pathway-discriminability-json",
        "--output-dir",
        output_dir
            .to_str()
            .ok_or("temporary output directory path is not valid UTF-8")?,
    ])?)
}

/// Build one deterministic synthetic scan-shard command.
fn smoke_single_config_shard_cli(
    data_dir: &Path,
    output_dir: &Path,
    shard_index: usize,
) -> Result<Cli, Box<dyn Error>> {
    let shard_index = shard_index.to_string();
    Ok(Cli::try_parse_from([
        "spectral-similarities-by-peaks",
        "scan-shard",
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
        "--shard-index",
        shard_index.as_str(),
        "--neighbors",
        "2",
        "--mz-tolerance",
        "0.05",
        "--histogram-bins",
        "3",
        "--pathway-representatives-per-class",
        "1",
        "--row-sample-size",
        "4",
        "--reference-sample-size",
        "6",
        "--max-spectra",
        "8",
        "--seed",
        "42",
    ])?)
}

/// Build the deterministic synthetic finalize-scan command for shard smoke tests.
fn smoke_single_config_finalize_cli(
    data_dir: &Path,
    output_dir: &Path,
) -> Result<Cli, Box<dyn Error>> {
    Ok(Cli::try_parse_from([
        "spectral-similarities-by-peaks",
        "finalize-scan",
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
        "--neighbors",
        "2",
        "--mz-tolerance",
        "0.05",
        "--histogram-bins",
        "3",
        "--pathway-representatives-per-class",
        "1",
        "--row-sample-size",
        "4",
        "--reference-sample-size",
        "6",
        "--max-spectra",
        "8",
        "--seed",
        "42",
    ])?)
}

/// Build the synthetic finalize-shard command for the sharded smoke test.
fn smoke_single_config_finalize_shard_cli(
    data_dir: &Path,
    output_dir: &Path,
    config_index: usize,
) -> Result<Cli, Box<dyn Error>> {
    let config_index = config_index.to_string();
    Ok(Cli::try_parse_from([
        "spectral-similarities-by-peaks",
        "finalize-shard",
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
        "--neighbors",
        "2",
        "--mz-tolerance",
        "0.05",
        "--histogram-bins",
        "3",
        "--pathway-representatives-per-class",
        "1",
        "--row-sample-size",
        "4",
        "--reference-sample-size",
        "6",
        "--max-spectra",
        "8",
        "--seed",
        "42",
        "--config-index",
        config_index.as_str(),
    ])?)
}

/// Build the synthetic finalize-merge command for the sharded smoke test.
fn smoke_single_config_finalize_merge_cli(
    data_dir: &Path,
    output_dir: &Path,
) -> Result<Cli, Box<dyn Error>> {
    Ok(Cli::try_parse_from([
        "spectral-similarities-by-peaks",
        "finalize-merge",
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
        "--neighbors",
        "2",
        "--mz-tolerance",
        "0.05",
        "--histogram-bins",
        "3",
        "--pathway-representatives-per-class",
        "1",
        "--row-sample-size",
        "4",
        "--reference-sample-size",
        "6",
        "--max-spectra",
        "8",
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
        stdout.contains("prefetch"),
        "missing prefetch command: {stdout}"
    );
    assert!(
        stdout.contains("scan-shard"),
        "missing scan-shard command: {stdout}"
    );
    assert!(
        stdout.contains("finalize-scan"),
        "missing finalize-scan command: {stdout}"
    );
    assert!(
        stdout.contains("render-pathway-artifacts"),
        "missing pathway artifact command: {stdout}"
    );
    Ok(())
}

#[test]
/// The prefetch subcommand loads the synthetic dataset without writing outputs.
fn prefetch_smoke_test_loads_dataset_cache() -> Result<(), Box<dyn Error>> {
    let root = smoke_root()?;
    let data_dir = root.join("data");
    fs::create_dir_all(&data_dir)?;

    let cli = Cli::try_parse_from([
        "spectral-similarities-by-peaks",
        "prefetch",
        "--dataset",
        "synthetic-smoke",
        "--data-dir",
        data_dir
            .to_str()
            .ok_or("temporary data directory path is not valid UTF-8")?,
    ])?;
    run::run(cli)?;

    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
/// The prefetch subcommand help is generated successfully.
fn prefetch_help_is_available() -> Result<(), Box<dyn Error>> {
    let Err(error) = Cli::try_parse_from(["spectral-similarities-by-peaks", "prefetch", "--help"])
    else {
        return Err(std::io::Error::other("prefetch help should short-circuit parsing").into());
    };
    assert_eq!(error.kind(), ErrorKind::DisplayHelp);
    let stdout = error.to_string();
    assert!(
        stdout.contains("--dataset"),
        "missing dataset flag: {stdout}"
    );
    assert!(
        stdout.contains("--gems-parts"),
        "missing GeMS parts flag: {stdout}"
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
/// The scan-shard subcommand help exposes shard selectors.
fn scan_shard_help_is_available() -> Result<(), Box<dyn Error>> {
    let Err(error) =
        Cli::try_parse_from(["spectral-similarities-by-peaks", "scan-shard", "--help"])
    else {
        return Err(std::io::Error::other("scan-shard help should short-circuit parsing").into());
    };
    assert_eq!(error.kind(), ErrorKind::DisplayHelp);
    let stdout = error.to_string();
    assert!(
        stdout.contains("--shard-index"),
        "missing shard-index flag: {stdout}"
    );
    assert!(
        stdout.contains("--peak-count"),
        "missing peak-count flag: {stdout}"
    );
    Ok(())
}

#[test]
/// The finalize-scan subcommand help is generated successfully.
fn finalize_scan_help_is_available() -> Result<(), Box<dyn Error>> {
    let Err(error) =
        Cli::try_parse_from(["spectral-similarities-by-peaks", "finalize-scan", "--help"])
    else {
        return Err(
            std::io::Error::other("finalize-scan help should short-circuit parsing").into(),
        );
    };
    assert_eq!(error.kind(), ErrorKind::DisplayHelp);
    let stdout = error.to_string();
    assert!(
        stdout.contains("--dataset"),
        "missing dataset flag: {stdout}"
    );
    assert!(
        stdout.contains("--similarity-config"),
        "missing config flag: {stdout}"
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

#[test]
/// Finalization reports missing distribution shards before writing final artifacts.
fn finalize_scan_requires_distribution_shards() -> Result<(), Box<dyn Error>> {
    let root = smoke_root()?;
    let data_dir = root.join("data");
    let output_dir = root.join("out");
    fs::create_dir_all(&data_dir)?;

    let cli = smoke_single_config_finalize_cli(&data_dir, &output_dir)?;
    let Err(error) = run::run(cli) else {
        return Err(std::io::Error::other("missing distribution shards should fail").into());
    };
    let message = error.to_string();
    assert!(
        message.contains("missing distribution checkpoint shards"),
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

/// Assert that a Parquet file exists and contains at least one row.
fn assert_parquet_nonempty(path: &Path) -> Result<(), Box<dyn Error>> {
    let metadata = fs::metadata(path)?;
    assert!(metadata.len() > 0, "{} is empty", path.display());

    let file = fs::File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let rows = reader.metadata().file_metadata().num_rows();
    assert!(rows > 0, "{} has no rows", path.display());
    Ok(())
}

/// Assert that the dense full-grid `NumPy` artifact has the expected axes.
fn assert_grid_npz_shapes(path: &Path) -> Result<(), Box<dyn Error>> {
    assert_grid_npz_shapes_with_configs(path, 18)
}

/// Assert that the dense full-grid `NumPy` artifact has the expected config axis.
fn assert_grid_npz_shapes_with_configs(
    path: &Path,
    expected_configs: usize,
) -> Result<(), Box<dyn Error>> {
    let file = fs::File::open(path)?;
    let mut reader = NpzReader::new(file)?;
    let peak_counts: Array1<u64> = reader.by_name("peak_counts.npy")?;
    let ks_statistic: Array3<f64> = reader.by_name("ks_statistic.npy")?;
    let ks_pvalue: Array3<f64> = reader.by_name("ks_pvalue_asymptotic.npy")?;
    let wasserstein: Array3<f64> = reader.by_name("wasserstein_1d.npy")?;
    let mean_delta: Array3<f64> = reader.by_name("mean_delta.npy")?;

    assert_eq!(peak_counts.len(), 128);
    assert_eq!(ks_statistic.shape(), &[expected_configs, 128, 128]);
    assert_eq!(ks_pvalue.shape(), &[expected_configs, 128, 128]);
    assert_eq!(wasserstein.shape(), &[expected_configs, 128, 128]);
    assert_eq!(mean_delta.shape(), &[expected_configs, 128, 128]);
    Ok(())
}

/// Assert that the pathway prediction dense-grid artifact has the expected axes.
fn assert_pathway_grid_npz_shapes(path: &Path) -> Result<(), Box<dyn Error>> {
    assert_pathway_grid_npz_shapes_with_configs(path, 18)
}

/// Assert that the pathway prediction dense-grid artifact has the expected config axis.
fn assert_pathway_grid_npz_shapes_with_configs(
    path: &Path,
    expected_configs: usize,
) -> Result<(), Box<dyn Error>> {
    let file = fs::File::open(path)?;
    let mut reader = NpzReader::new(file)?;
    let peak_counts: Array1<u64> = reader.by_name("peak_counts.npy")?;
    let total_variation: Array3<f64> = reader.by_name("total_variation.npy")?;
    let jensen_shannon: Array3<f64> = reader.by_name("jensen_shannon_distance.npy")?;
    let hellinger: Array3<f64> = reader.by_name("hellinger_distance.npy")?;

    assert_eq!(peak_counts.len(), 128);
    assert_eq!(total_variation.shape(), &[expected_configs, 128, 128]);
    assert_eq!(jensen_shannon.shape(), &[expected_configs, 128, 128]);
    assert_eq!(hellinger.shape(), &[expected_configs, 128, 128]);
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
            "mean_delta_linear",
            "mean_delta_log",
            "ks_statistic_linear",
            "ks_statistic_log",
            "ks_pvalue_asymptotic_linear",
            "ks_pvalue_asymptotic_log",
            "wasserstein_1d_linear",
            "wasserstein_1d_log",
        ] {
            let stem = output_dir.join("heatmaps").join(config).join(metric);
            assert_svg_artifact(&stem.with_extension("svg"))?;
            assert_png_artifact(&stem.with_extension("png"))?;
        }
    }
    Ok(())
}

/// Assert that one config's distribution heatmaps were written in SVG and PNG form.
fn assert_single_config_heatmap_artifacts(
    output_dir: &Path,
    config: &str,
) -> Result<(), Box<dyn Error>> {
    for metric in [
        "mean_delta_linear",
        "mean_delta_log",
        "ks_statistic_linear",
        "ks_statistic_log",
        "ks_pvalue_asymptotic_linear",
        "ks_pvalue_asymptotic_log",
        "wasserstein_1d_linear",
        "wasserstein_1d_log",
    ] {
        let stem = output_dir.join("heatmaps").join(config).join(metric);
        assert_svg_artifact(&stem.with_extension("svg"))?;
        assert_png_artifact(&stem.with_extension("png"))?;
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

/// Assert that one config's pathway prediction plots were written.
fn assert_single_config_pathway_prediction_artifacts(
    output_dir: &Path,
    config: &str,
) -> Result<(), Box<dyn Error>> {
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
    Ok(())
}

/// Assert that the AUROC / AUPRC line plots from render-pathway-discriminability
/// were written in both SVG and PNG form.
fn assert_pathway_discriminability_artifacts(output_dir: &Path) -> Result<(), Box<dyn Error>> {
    for metric in ["auroc", "auprc"] {
        let stem = output_dir
            .join("pathway_discriminability_plots")
            .join(metric);
        assert_svg_artifact(&stem.with_extension("svg"))?;
        assert_png_artifact(&stem.with_extension("png"))?;
    }
    Ok(())
}

/// Assert that `pathway_discriminability_lines.json` is present and that
/// its top-level shape matches the contract consumed by the WASM viewer.
fn assert_pathway_discriminability_json(output_dir: &Path) -> Result<(), Box<dyn Error>> {
    let path = output_dir.join("pathway_discriminability_lines.json");
    let text = fs::read_to_string(&path)?;
    let document: serde_json::Value = serde_json::from_str(&text)?;
    let object = document
        .as_object()
        .ok_or("pathway_discriminability_lines.json root is not an object")?;
    for key in ["peak_counts", "configs", "pathways"] {
        assert!(
            object.contains_key(key),
            "missing top-level key `{key}` in {}",
            path.display()
        );
    }
    let pathways = object["pathways"]
        .as_array()
        .ok_or("`pathways` is not a JSON array")?;
    let first = pathways
        .first()
        .ok_or("`pathways` array is empty")?
        .as_object()
        .ok_or("first `pathways` entry is not a JSON object")?;
    let label = first["label"]
        .as_str()
        .ok_or("first `pathways` entry has no string `label`")?;
    assert_eq!(
        label, "Aggregate (micro-averaged)",
        "first pathway entry should be the aggregate micro-averaged classifier"
    );
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

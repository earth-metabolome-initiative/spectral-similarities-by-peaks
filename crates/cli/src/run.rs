//! Command execution for the experiment binary.

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use arrow_array::{RecordBatch, StringArray, UInt64Array};
use mass_spectrometry::prelude::GenericSpectrum;
use ndarray::{Array1, Array3};
use ndarray_npy::NpzReader;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;

use crate::{
    checkpoint::{self, CheckpointBase, RunFingerprint},
    cli::{
        Cli, Commands, FinalizeMergeArgs, FinalizeScanArgs, FinalizeShardArgs, PrefetchArgs,
        RenderHeatmapArgs, RenderPathwayArtifactArgs, ScanArgs, ScanShardArgs,
    },
    data::{self, load_dataset_records, load_records},
    distribution::{
        compare_distributions, histogram_sorted_distribution, self_comparison,
        summarize_sorted_distribution,
    },
    model::{
        DistributionSummary, LoadedRecord, PEAK_COUNT_GRID_SIZE, ScoreDistribution,
        SimilarityConfig,
    },
    neighbors::{SearchBatch, compute_neighbors},
    output::{self, GridArrays, OutputWriters},
    pathway::{pathway_labels, pathway_representative_indices, score_pathway_representatives},
    pathway_artifacts::{
        write_existing_pathway_prediction_artifacts, write_pathway_prediction_artifacts,
    },
    progress::{ProgressTask, ScanProgress},
    spectra::{prepare_spectra, select_query_ids, select_reference_ids},
    visualize::write_heatmaps,
};

/// Dispatch parsed command-line arguments to the selected command.
///
/// # Errors
///
/// Returns an error when command validation fails, input datasets cannot be
/// loaded, spectra cannot be prepared, or output artifacts cannot be
/// written.
pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Prefetch(args) => run_prefetch(&args),
        Commands::Scan(args) => run_scan(args),
        Commands::ScanShard(args) => run_scan_shard(args),
        Commands::FinalizeScan(args) => run_finalize_scan(args),
        Commands::FinalizeShard(args) => run_finalize_shard(args),
        Commands::FinalizeMerge(args) => run_finalize_merge(args),
        Commands::RenderHeatmaps(args) => run_render_heatmaps(args),
        Commands::RenderPathwayArtifacts(args) => run_render_pathway_artifacts(&args),
        Commands::ComputePathwayDiscriminability(args) => {
            run_compute_pathway_discriminability(&args)
        }
        Commands::ComputeConfigDiversity(args) => run_compute_config_diversity(&args),
        Commands::ReEncodeParquets(args) => run_re_encode_parquets(&args),
    }
}

/// Download and cache the selected dataset without computing similarities.
fn run_prefetch(args: &PrefetchArgs) -> Result<()> {
    let progress = ScanProgress::new();
    let load_progress = progress.spinner(format!("prefetching {} records", args.dataset.as_str()));
    let records = load_dataset_records(args.dataset, &args.data_dir, args.gems_parts.as_deref())?;
    load_progress.finish();
    println!(
        "Prefetched {} {} records into {}",
        records.len(),
        args.dataset.as_str(),
        args.data_dir.display()
    );
    Ok(())
}

/// Re-render heatmap artifacts from an existing scan output directory.
fn run_render_heatmaps(mut args: RenderHeatmapArgs) -> Result<()> {
    args.validate()?;
    let progress = ScanProgress::new();
    let arrays = read_grid_npz(&args.output_dir)?;
    let configs = read_grid_configs(&args.output_dir)?;
    write_heatmaps(
        &args.output_dir,
        &configs,
        &arrays,
        &args.threshold_alphas,
        &progress,
    )
}

/// Rebuild pathway prediction artifacts from an existing scan output directory.
fn run_render_pathway_artifacts(args: &RenderPathwayArtifactArgs) -> Result<()> {
    let progress = ScanProgress::new();
    write_existing_pathway_prediction_artifacts(&args.output_dir, &progress)
}

/// Compute AUROC / AUPRC of pathway-pair scores from an existing scan output.
fn run_compute_pathway_discriminability(
    args: &crate::cli::ComputePathwayDiscriminabilityArgs,
) -> Result<()> {
    let progress = ScanProgress::new();
    crate::pathway_discriminability::write_pathway_discriminability(&args.output_dir, &progress)
}

/// Compute per-config mean KS statistic and rank configs by "diversity".
fn run_compute_config_diversity(args: &crate::cli::ComputeConfigDiversityArgs) -> Result<()> {
    let progress = ScanProgress::new();
    crate::config_diversity::write_config_diversity(&args.output_dir, &progress)
}

/// Walk every `.parquet` under `args.output_dir` and rewrite it in place
/// using this crate's default zstd compression.
///
/// Existing artifacts produced by older versions of the binary used the
/// parquet crate's default (Snappy) codec; a 10-shard sample of
/// `pathway_scores.parquet` showed an 85.8 % size reduction when re-encoded
/// at zstd-22. This subcommand applies the same in-place re-encoding via
/// streaming reads + writes so the result is still valid `.parquet`
/// (random columnar access preserved, unlike a bare `zstd --rm` wrap).
fn run_re_encode_parquets(args: &crate::cli::ReEncodeParquetsArgs) -> Result<()> {
    use rayon::prelude::*;

    let root = &args.output_dir;
    if !root.is_dir() {
        anyhow::bail!("{} is not a directory", root.display());
    }
    let paths = collect_parquets(root)?;
    if paths.is_empty() {
        return Ok(());
    }
    let progress = ScanProgress::new();
    let task = progress.bar(
        u64::try_from(paths.len()).unwrap_or(u64::MAX),
        "re-encoding parquets",
    );
    paths.par_iter().try_for_each(|path| -> Result<()> {
        re_encode_parquet(path)?;
        task.inc(1);
        Ok(())
    })?;
    task.finish();
    Ok(())
}

/// Recursively gather every `.parquet` file under `root` (sorted).
fn collect_parquets(root: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
            let entry = entry.with_context(|| format!("listing {}", dir.display()))?;
            let entry_path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                stack.push(entry_path);
            } else if file_type.is_file()
                && entry_path.extension().is_some_and(|ext| ext == "parquet")
            {
                paths.push(entry_path);
            }
        }
    }
    paths.sort();
    Ok(paths)
}

/// Read a parquet file batch-by-batch and rewrite it in place using the
/// crate's default `WriterProperties`. Streaming so per-file memory stays
/// bounded by a single record batch.
fn re_encode_parquet(path: &Path) -> Result<()> {
    use arrow_array::RecordBatchReader;
    use parquet::arrow::ArrowWriter;

    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading metadata from {}", path.display()))?
        .build()
        .with_context(|| format!("building reader for {}", path.display()))?;
    let schema = reader.schema();

    let temp = path.with_extension("parquet.zstd-tmp");
    let tmp_file =
        fs::File::create(&temp).with_context(|| format!("creating {}", temp.display()))?;
    let mut writer = ArrowWriter::try_new(
        tmp_file,
        schema,
        Some(crate::output::parquet_writer_props()),
    )
    .with_context(|| format!("opening writer for {}", temp.display()))?;
    for batch in reader {
        let batch = batch.with_context(|| format!("decoding batch in {}", path.display()))?;
        writer
            .write(&batch)
            .with_context(|| format!("writing batch to {}", temp.display()))?;
    }
    writer
        .close()
        .with_context(|| format!("finalizing {}", temp.display()))?;
    fs::rename(&temp, path)
        .with_context(|| format!("replacing {} with re-encoded copy", path.display()))?;
    Ok(())
}

/// Read dense grid matrices from an existing `distribution_grid.npz` artifact.
fn read_grid_npz(output_dir: &Path) -> Result<GridArrays> {
    let path = output_dir.join("distribution_grid.npz");
    let file = fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = NpzReader::new(file).with_context(|| format!("reading {}", path.display()))?;
    let peak_counts: Array1<u64> = reader.by_name("peak_counts.npy")?;
    let mean_delta: Array3<f64> = reader.by_name("mean_delta.npy")?;
    let ks_statistic: Array3<f64> = reader.by_name("ks_statistic.npy")?;
    let ks_pvalue_asymptotic: Array3<f64> = reader.by_name("ks_pvalue_asymptotic.npy")?;
    let wasserstein_1d: Array3<f64> = reader.by_name("wasserstein_1d.npy")?;
    Ok(GridArrays {
        peak_counts,
        mean_delta,
        ks_statistic,
        ks_pvalue_asymptotic,
        wasserstein_1d,
    })
}

/// Read heatmap config labels from `distribution_grid_configs.parquet`.
fn read_grid_configs(output_dir: &Path) -> Result<Vec<String>> {
    let path = output_dir.join("distribution_grid_configs.parquet");
    let file = fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("reading metadata from {}", path.display()))?
        .build()
        .with_context(|| format!("building reader for {}", path.display()))?;

    let mut configs = Vec::<(usize, String)>::new();
    for batch in reader {
        read_grid_config_batch(&batch?, &mut configs)?;
    }
    configs.sort_by_key(|(config_index, _config)| *config_index);
    let configs = configs
        .into_iter()
        .map(|(_config_index, config)| config)
        .collect::<Vec<_>>();
    if configs.is_empty() {
        bail!("{} contains no heatmap configs", path.display());
    }
    Ok(configs)
}

/// Append grid config rows from one `Arrow` record batch.
fn read_grid_config_batch(batch: &RecordBatch, configs: &mut Vec<(usize, String)>) -> Result<()> {
    let config_indices = required_column::<UInt64Array>(batch, "config_index")?;
    let config_names = required_column::<StringArray>(batch, "config")?;
    for row in 0..batch.num_rows() {
        let config_index = usize::try_from(config_indices.value(row))
            .context("config_index does not fit usize")?;
        configs.push((config_index, config_names.value(row).to_string()));
    }
    Ok(())
}

/// Return a typed required `Arrow` column from a record batch.
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

/// Execute a full similarity scan and write all artifacts.
fn run_scan(mut args: ScanArgs) -> Result<()> {
    args.validate()?;
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("creating {}", args.output_dir.display()))?;

    let progress = ScanProgress::new();
    let data = load_scan_data(&args, &progress)?;
    let writer_progress = progress.spinner("opening output writers");
    let mut writers = OutputWriters::create(&args.output_dir)?;
    writer_progress.finish();
    let total_peak_runs = args
        .similarity_configs
        .len()
        .saturating_mul(PEAK_COUNT_GRID_SIZE);
    let scan_progress = progress.bar(
        u64::try_from(total_peak_runs).unwrap_or(u64::MAX),
        "scanning peak-count grid",
    );

    let inputs = ScanInputs {
        args: &args,
        progress: &progress,
        records: &data.records,
        query_ids: &data.query_ids,
        reference_ids: &data.reference_ids,
        checkpoint_base: &data.checkpoint_base,
        scan_progress: &scan_progress,
    };
    for config in &args.similarity_configs {
        run_similarity_config(&inputs, &mut writers, config)?;
    }
    scan_progress.finish();

    writers.finish(&args.threshold_alphas, &progress)?;
    if args.pathway_representatives_per_class > 0 {
        write_pathway_prediction_artifacts(&args.output_dir, &progress)?;
    }
    Ok(())
}

/// Loaded dataset state shared by local scans, shard jobs, and finalization.
struct ScanData {
    /// Loaded spectrum records after optional truncation.
    records: Vec<LoadedRecord>,
    /// Deterministically selected query row ids.
    query_ids: Vec<usize>,
    /// Deterministically selected fixed reference row ids.
    reference_ids: Vec<usize>,
    /// Scan-level checkpoint fingerprint base.
    checkpoint_base: CheckpointBase,
}

/// Load records and derive deterministic sampling state for a scan command.
fn load_scan_data(args: &ScanArgs, progress: &ScanProgress) -> Result<ScanData> {
    let load_progress = progress.spinner(format!("loading {} records", args.dataset.as_str()));
    let mut records = load_records(args)?;
    load_progress.finish();
    if let Some(max_spectra) = args.max_spectra {
        records.truncate(max_spectra.min(records.len()));
    }
    if records.is_empty() {
        bail!("no spectra loaded");
    }
    ensure_pathway_labels_available(args, &records)?;

    let query_ids = select_query_ids(records.len(), args.row_sample_size, args.seed);
    let reference_ids = select_reference_ids(records.len(), args.reference_sample_size, args.seed);
    let checkpoint_base = CheckpointBase::new(args, &records, &query_ids, &reference_ids);

    let mut keep: BTreeSet<usize> = BTreeSet::new();
    keep.extend(query_ids.iter().copied());
    keep.extend(reference_ids.iter().copied());
    if args.pathway_representatives_per_class > 0 {
        keep.extend(pathway_representative_indices(
            &records,
            args.pathway_representatives_per_class,
        ));
    }

    if keep.len() < records.len() {
        let subset_progress = progress.spinner(format!(
            "subsetting records to sampled working set ({} of {})",
            keep.len(),
            records.len()
        ));
        let records = data::subset_records(records, &keep);
        let query_ids = data::remap_sorted_ids(&query_ids, &keep);
        let reference_ids = data::remap_sorted_ids(&reference_ids, &keep);
        subset_progress.finish();
        return Ok(ScanData {
            records,
            query_ids,
            reference_ids,
            checkpoint_base,
        });
    }

    Ok(ScanData {
        records,
        query_ids,
        reference_ids,
        checkpoint_base,
    })
}

/// Execute one restartable score-distribution shard.
fn run_scan_shard(mut args: ScanShardArgs) -> Result<()> {
    args.scan.validate()?;
    let assignment = select_shard_assignment(&args)?;
    fs::create_dir_all(&args.scan.output_dir)
        .with_context(|| format!("creating {}", args.scan.output_dir.display()))?;

    let progress = ScanProgress::new();
    let mut data = load_scan_data(&args.scan, &progress)?;
    let scan_progress = progress.bar(
        1,
        format!(
            "scanning shard {} config {} top {}",
            assignment.shard_index, assignment.config_index, assignment.peak_count
        ),
    );
    let config_name = assignment.config.name();
    let fingerprint =
        data.checkpoint_base
            .fingerprint(&args.scan, &assignment.config, &config_name);
    run_peak_count_shard(
        &args.scan,
        &progress,
        &scan_progress,
        &mut data,
        &assignment.config,
        &config_name,
        &fingerprint,
        assignment.peak_count,
    )?;
    scan_progress.inc(1);
    scan_progress.finish();
    Ok(())
}

/// Finalize global artifacts from completed shard checkpoints.
fn run_finalize_scan(mut args: FinalizeScanArgs) -> Result<()> {
    args.scan.validate()?;
    fs::create_dir_all(&args.scan.output_dir)
        .with_context(|| format!("creating {}", args.scan.output_dir.display()))?;
    ensure_expected_distribution_checkpoint_paths(&args.scan)?;
    ensure_expected_pathway_shard_paths(&args.scan)?;

    let progress = ScanProgress::new();
    let data = load_scan_data(&args.scan, &progress)?;
    let writer_progress = progress.spinner("opening output writers");
    let mut writers = OutputWriters::create(&args.scan.output_dir)?;
    writer_progress.finish();
    let scan_progress = progress.bar(
        u64::try_from(
            args.scan
                .similarity_configs
                .len()
                .saturating_mul(PEAK_COUNT_GRID_SIZE),
        )
        .unwrap_or(u64::MAX),
        "finalizing peak-count grid",
    );
    let inputs = ScanInputs {
        args: &args.scan,
        progress: &progress,
        records: &data.records,
        query_ids: &data.query_ids,
        reference_ids: &data.reference_ids,
        checkpoint_base: &data.checkpoint_base,
        scan_progress: &scan_progress,
    };

    for config in &args.scan.similarity_configs {
        finalize_similarity_config(&inputs, &mut writers, config)?;
    }
    scan_progress.finish();

    writers.finish(&args.scan.threshold_alphas, &progress)?;
    if args.scan.pathway_representatives_per_class > 0 {
        write_pathway_prediction_artifacts(&args.scan.output_dir, &progress)?;
    }
    Ok(())
}

/// Build per-config finalize artifacts for one similarity configuration.
///
/// # Errors
///
/// Returns an error if validation fails, the requested config index is
/// out-of-range, expected distribution or pathway shards for the config are
/// missing, or finalize processing fails.
fn run_finalize_shard(mut args: FinalizeShardArgs) -> Result<()> {
    args.scan.validate()?;
    if args.config_index >= args.scan.similarity_configs.len() {
        bail!(
            "--config-index {} is outside the selected configs (len {})",
            args.config_index,
            args.scan.similarity_configs.len()
        );
    }
    let config = args.scan.similarity_configs[args.config_index].clone();
    let single = [config.clone()];
    ensure_distribution_checkpoint_paths(&args.scan, &single)?;
    ensure_pathway_shard_paths(&args.scan, &single)?;
    fs::create_dir_all(&args.scan.output_dir)
        .with_context(|| format!("creating {}", args.scan.output_dir.display()))?;

    let progress = ScanProgress::new();
    let data = load_scan_data(&args.scan, &progress)?;
    let config_name = config.name();
    let writer_progress =
        progress.spinner(format!("opening output writers for {config_name} shard"));
    let mut writers = OutputWriters::create_for_shard(&args.scan.output_dir, &config_name)?;
    writer_progress.finish();
    let scan_progress = progress.bar(
        u64::try_from(PEAK_COUNT_GRID_SIZE).unwrap_or(u64::MAX),
        format!("finalizing {config_name} shard"),
    );
    let inputs = ScanInputs {
        args: &args.scan,
        progress: &progress,
        records: &data.records,
        query_ids: &data.query_ids,
        reference_ids: &data.reference_ids,
        checkpoint_base: &data.checkpoint_base,
        scan_progress: &scan_progress,
    };

    finalize_similarity_config(&inputs, &mut writers, &config)?;
    scan_progress.finish();

    let canonical_dir = args.scan.output_dir.clone();
    writers.finish_with_mode(
        output::FinishMode::PerConfigShard {
            canonical_dir: &canonical_dir,
        },
        &args.scan.threshold_alphas,
        &progress,
    )
}

/// Concatenate per-config shard outputs into the canonical finalize artifacts.
///
/// # Errors
///
/// Returns an error when any per-config shard directory is missing, Parquet
/// concatenation fails, or the trailing pathway-prediction artifact build
/// fails.
fn run_finalize_merge(mut args: FinalizeMergeArgs) -> Result<()> {
    args.scan.validate()?;
    fs::create_dir_all(&args.scan.output_dir)
        .with_context(|| format!("creating {}", args.scan.output_dir.display()))?;

    let progress = ScanProgress::new();
    let config_names: Vec<String> = args
        .scan
        .similarity_configs
        .iter()
        .map(SimilarityConfig::name)
        .collect();

    let check_progress = progress.spinner("validating per-config shard outputs");
    ensure_per_config_shard_outputs(
        &args.scan.output_dir,
        &config_names,
        args.scan.pathway_representatives_per_class > 0,
    )?;
    check_progress.finish();

    let merge_progress = progress.spinner("concatenating per-config Parquet outputs");
    output::merge_per_config_parquets(
        &args.scan.output_dir,
        &config_names,
        args.scan.pathway_representatives_per_class > 0,
    )?;
    merge_progress.finish();

    let grid_progress = progress.spinner("stacking grid-matrix slices into distribution_grid.npz");
    output::merge_grid_matrix_slices(&args.scan.output_dir, &config_names)?;
    grid_progress.finish();

    if args.scan.pathway_representatives_per_class > 0 {
        write_pathway_prediction_artifacts(&args.scan.output_dir, &progress)?;
    }

    if !args.keep_shard_dir {
        let cleanup_progress = progress.spinner("removing _finalize_shards/");
        let shard_root = args.scan.output_dir.join(output::FINALIZE_SHARD_DIR);
        if shard_root.is_dir() {
            fs::remove_dir_all(&shard_root)
                .with_context(|| format!("removing {}", shard_root.display()))?;
        }
        cleanup_progress.finish();
    }

    Ok(())
}

/// Verify every per-config shard wrote the expected outputs before merging.
fn ensure_per_config_shard_outputs(
    output_dir: &Path,
    config_names: &[String],
    include_pathway_outputs: bool,
) -> Result<()> {
    let mut missing: Vec<String> = Vec::new();
    let mut required: Vec<&str> = vec![
        "distribution_summary.parquet",
        "distribution_histograms.parquet",
        "distribution_tests.parquet",
        "distribution_grid.parquet",
        "grid_matrix.bincode.zst",
    ];
    if include_pathway_outputs {
        required.push("pathway_scores.parquet");
        required.push("pathway_predictions.parquet");
    }
    for config_name in config_names {
        let shard_dir = output::shard_directory(output_dir, config_name);
        for file_name in &required {
            let path = shard_dir.join(file_name);
            if !path.is_file() {
                missing.push(format!("{config_name}/{file_name}"));
            }
        }
    }
    if missing.is_empty() {
        Ok(())
    } else {
        bail!(
            "missing per-config shard outputs under {}/_finalize_shards: {}",
            output_dir.display(),
            missing.join(", ")
        );
    }
}

/// One selected shard from the two-dimensional config/peak grid.
struct ShardAssignment {
    /// Similarity configuration to evaluate.
    config: SimilarityConfig,
    /// One-based retained peak count to evaluate.
    peak_count: usize,
    /// Zero-based similarity configuration index.
    config_index: usize,
    /// Zero-based shard index over the full selected grid.
    shard_index: usize,
}

/// Resolve explicit or indexed shard arguments to one config/peak assignment.
fn select_shard_assignment(args: &ScanShardArgs) -> Result<ShardAssignment> {
    match (args.peak_count, args.shard_index) {
        (Some(_), Some(_)) => {
            bail!("scan-shard accepts either --peak-count or --shard-index, not both");
        }
        (Some(peak_count), None) => {
            validate_peak_count(peak_count)?;
            if args.scan.similarity_configs.len() != 1 {
                bail!(
                    "scan-shard with --peak-count requires exactly one --similarity-config, got {}",
                    args.scan.similarity_configs.len()
                );
            }
            Ok(ShardAssignment {
                config: args.scan.similarity_configs[0].clone(),
                peak_count,
                config_index: 0,
                shard_index: peak_count - 1,
            })
        }
        (None, Some(shard_index)) => {
            let configs = args.scan.similarity_configs.len();
            let total_shards = configs.saturating_mul(PEAK_COUNT_GRID_SIZE);
            if shard_index >= total_shards {
                bail!(
                    "--shard-index {shard_index} is outside the selected grid of {total_shards} shards"
                );
            }
            let config_index = shard_index / PEAK_COUNT_GRID_SIZE;
            let peak_count = shard_index % PEAK_COUNT_GRID_SIZE + 1;
            Ok(ShardAssignment {
                config: args.scan.similarity_configs[config_index].clone(),
                peak_count,
                config_index,
                shard_index,
            })
        }
        (None, None) => {
            bail!("scan-shard requires either --peak-count or --shard-index");
        }
    }
}

/// Validate one retained peak count from the fixed experiment grid.
fn validate_peak_count(peak_count: usize) -> Result<()> {
    if !(1..=PEAK_COUNT_GRID_SIZE).contains(&peak_count) {
        bail!("--peak-count must be in 1..={PEAK_COUNT_GRID_SIZE}, got {peak_count}");
    }
    Ok(())
}

/// Fail early when finalization is missing expected distribution checkpoint files.
fn ensure_expected_distribution_checkpoint_paths(args: &ScanArgs) -> Result<()> {
    ensure_distribution_checkpoint_paths(args, &args.similarity_configs)
}

/// Fail early when any of the supplied configs is missing a distribution shard.
fn ensure_distribution_checkpoint_paths(
    args: &ScanArgs,
    configs: &[SimilarityConfig],
) -> Result<()> {
    let missing = configs
        .iter()
        .flat_map(|config| {
            let config_name = config.name();
            let checked_config_name = config_name.clone();
            (1..=PEAK_COUNT_GRID_SIZE)
                .filter(move |&peak_count| {
                    !checkpoint::checkpoint_exists(
                        &args.output_dir,
                        &checked_config_name,
                        peak_count,
                    )
                })
                .map(move |peak_count| format!("{config_name}/top_{peak_count:03}.bincode.zst"))
        })
        .take(20)
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "missing distribution checkpoint shards under {}: {}{}",
            args.output_dir.join("distributions").display(),
            missing.join(", "),
            if missing.len() == 20 { ", ..." } else { "" }
        );
    }
    Ok(())
}

/// Fail early when finalization is missing expected pathway shard files.
fn ensure_expected_pathway_shard_paths(args: &ScanArgs) -> Result<()> {
    ensure_pathway_shard_paths(args, &args.similarity_configs)
}

/// Fail early when any of the supplied configs is missing a pathway shard.
fn ensure_pathway_shard_paths(args: &ScanArgs, configs: &[SimilarityConfig]) -> Result<()> {
    if args.pathway_representatives_per_class == 0 {
        return Ok(());
    }
    let missing = configs
        .iter()
        .flat_map(|config| {
            let config_name = config.name();
            let checked_config_name = config_name.clone();
            (1..=PEAK_COUNT_GRID_SIZE)
                .filter(move |&peak_count| {
                    !output::pathway_shard_exists(
                        &args.output_dir,
                        &checked_config_name,
                        peak_count,
                    )
                })
                .map(move |peak_count| format!("{config_name}/top_{peak_count:03}"))
        })
        .take(20)
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "missing pathway prediction shards under {}: {}{}",
            args.output_dir.join("pathway_shards").display(),
            missing.join(", "),
            if missing.len() == 20 { ", ..." } else { "" }
        );
    }
    Ok(())
}

/// Ensure pathway scoring was not requested on records without pathway labels.
fn ensure_pathway_labels_available(args: &ScanArgs, records: &[LoadedRecord]) -> Result<()> {
    if args.pathway_representatives_per_class == 0 {
        return Ok(());
    }
    let labeled_records = records
        .iter()
        .filter(|record| !pathway_labels(record.npc_pathway.as_deref()).is_empty())
        .count();
    if labeled_records == 0 {
        bail!(
            "pathway scoring was requested with --pathway-representatives-per-class {}, but no loaded records contain NPC_PATHWAYS labels",
            args.pathway_representatives_per_class
        );
    }
    Ok(())
}

/// Shared immutable inputs for one scan.
struct ScanInputs<'a> {
    /// Parsed scan arguments.
    args: &'a ScanArgs,
    /// Shared progress reporter.
    progress: &'a ScanProgress,
    /// Loaded spectrum records.
    records: &'a [LoadedRecord],
    /// Query row ids selected for the run.
    query_ids: &'a [usize],
    /// Reference row ids selected for the run.
    reference_ids: &'a [usize],
    /// Shared checkpoint fingerprint fields.
    checkpoint_base: &'a CheckpointBase,
    /// Outer peak-grid progress bar.
    scan_progress: &'a ProgressTask,
}

/// Run all peak counts and distribution comparisons for one similarity config.
fn run_similarity_config(
    inputs: &ScanInputs<'_>,
    writers: &mut OutputWriters,
    config: &SimilarityConfig,
) -> Result<()> {
    let config_name = config.name();
    let fingerprint = inputs
        .checkpoint_base
        .fingerprint(inputs.args, config, &config_name);
    let mut distributions = Vec::with_capacity(PEAK_COUNT_GRID_SIZE);
    for peak_count in 1..=PEAK_COUNT_GRID_SIZE {
        let distribution = run_peak_count(
            inputs,
            writers,
            config,
            &config_name,
            &fingerprint,
            peak_count,
        )?;
        distributions.push(distribution);
        inputs.scan_progress.inc(1);
    }

    write_adjacent_comparisons(inputs, writers, config, &config_name, &distributions)?;
    write_grid_comparisons(inputs, writers, config, &config_name, &distributions)
}

/// Load completed checkpoints for one config and write final artifacts.
fn finalize_similarity_config(
    inputs: &ScanInputs<'_>,
    writers: &mut OutputWriters,
    config: &SimilarityConfig,
) -> Result<()> {
    let config_name = config.name();
    let fingerprint = inputs
        .checkpoint_base
        .fingerprint(inputs.args, config, &config_name);
    let mut distributions = Vec::with_capacity(PEAK_COUNT_GRID_SIZE);
    for peak_count in 1..=PEAK_COUNT_GRID_SIZE {
        inputs
            .scan_progress
            .set_message(format!("finalizing {config_name} top {peak_count} peaks"));
        let distribution = checkpoint::load_distribution(
            &inputs.args.output_dir,
            inputs.args.dataset.as_str(),
            &config_name,
            config.metric_label(),
            peak_count,
            &fingerprint,
        )
        .with_context(|| {
            format!(
                "missing or invalid checkpoint for {config_name} top {peak_count} peaks at {}",
                checkpoint::checkpoint_path(&inputs.args.output_dir, &config_name, peak_count)
                    .display()
            )
        })?;
        let summary =
            summarize_sorted_distribution(inputs.args, config, peak_count, &distribution.scores)?;
        let histogram =
            histogram_sorted_distribution(inputs.args, config, peak_count, &distribution.scores)?;
        writers.write_summary(&summary)?;
        writers.write_histogram(&histogram)?;
        if inputs.args.pathway_representatives_per_class > 0 {
            writers.write_pathway_shard(&inputs.args.output_dir, &config_name, peak_count)?;
        }
        distributions.push(distribution);
        inputs.scan_progress.inc(1);
    }

    write_adjacent_comparisons(inputs, writers, config, &config_name, &distributions)?;
    write_grid_comparisons(inputs, writers, config, &config_name, &distributions)
}

/// Run one retained-peak count as a shard-safe checkpoint-only computation.
#[allow(clippy::too_many_arguments)]
fn run_peak_count_shard(
    args: &ScanArgs,
    progress: &ScanProgress,
    scan_progress: &ProgressTask,
    data: &mut ScanData,
    config: &SimilarityConfig,
    config_name: &str,
    fingerprint: &RunFingerprint,
    peak_count: usize,
) -> Result<()> {
    scan_progress.set_message(format!("scanning {config_name} top {peak_count} peaks"));
    let cached_distribution = checkpoint::load_distribution(
        &args.output_dir,
        args.dataset.as_str(),
        config_name,
        config.metric_label(),
        peak_count,
        fingerprint,
    );
    let needs_pathway_shard = args.pathway_representatives_per_class > 0
        && !output::pathway_shard_exists(&args.output_dir, config_name, peak_count);
    if cached_distribution.is_some() && !needs_pathway_shard {
        scan_progress.set_message(format!("using cached {config_name} top {peak_count} peaks"));
        return Ok(());
    }

    let spectra = prepare_spectra(
        progress,
        &data.records,
        peak_count,
        args.mz_tolerance,
        !args.no_merge_close_peaks,
    )
    .with_context(|| format!("preparing spectra for top {peak_count} peaks"))?;

    data::drop_record_spectra(&mut data.records)
        .context("releasing source spectra after shard prepare")?;

    let inputs = ScanInputs {
        args,
        progress,
        records: &data.records,
        query_ids: &data.query_ids,
        reference_ids: &data.reference_ids,
        checkpoint_base: &data.checkpoint_base,
        scan_progress,
    };

    if cached_distribution.is_none() {
        let (distribution, _summary) =
            compute_score_distribution(&inputs, config, config_name, peak_count, &spectra)?;
        checkpoint::store_distribution(
            &args.output_dir,
            args.dataset.as_str(),
            config_name,
            config.metric_label(),
            &distribution,
            fingerprint,
        )
        .with_context(|| {
            format!("storing {config_name} top {peak_count} distribution checkpoint")
        })?;
    }

    if needs_pathway_shard {
        if let Some((pathway_scores, pathway_predictions)) = score_pathway_representatives(
            args,
            progress,
            config,
            peak_count,
            inputs.records,
            &spectra,
            inputs.query_ids,
        )? {
            output::write_pathway_shard(
                &args.output_dir,
                config_name,
                peak_count,
                &pathway_scores,
                &pathway_predictions,
            )?;
        }
    }
    Ok(())
}

/// Run one retained-peak count for one similarity config.
fn run_peak_count(
    inputs: &ScanInputs<'_>,
    writers: &mut OutputWriters,
    config: &SimilarityConfig,
    config_name: &str,
    fingerprint: &RunFingerprint,
    peak_count: usize,
) -> Result<ScoreDistribution> {
    inputs
        .scan_progress
        .set_message(format!("scanning {config_name} top {peak_count} peaks"));
    if let Some(distribution) = checkpoint::load_distribution(
        &inputs.args.output_dir,
        inputs.args.dataset.as_str(),
        config_name,
        config.metric_label(),
        peak_count,
        fingerprint,
    ) {
        inputs
            .scan_progress
            .set_message(format!("using cached {config_name} top {peak_count} peaks"));
        write_cached_distribution_outputs(inputs, writers, config, &distribution).with_context(
            || format!("writing cached outputs for {config_name} top {peak_count} peaks"),
        )?;
        return Ok(distribution);
    }

    let spectra = prepare_spectra(
        inputs.progress,
        inputs.records,
        peak_count,
        inputs.args.mz_tolerance,
        !inputs.args.no_merge_close_peaks,
    )
    .with_context(|| format!("preparing spectra for top {peak_count} peaks"))?;

    let (distribution, summary) =
        compute_score_distribution(inputs, config, config_name, peak_count, &spectra)?;
    checkpoint::store_distribution(
        &inputs.args.output_dir,
        inputs.args.dataset.as_str(),
        config_name,
        config.metric_label(),
        &distribution,
        fingerprint,
    )
    .with_context(|| format!("storing {config_name} top {peak_count} distribution checkpoint"))?;
    let histogram =
        histogram_sorted_distribution(inputs.args, config, peak_count, &distribution.scores)?;
    writers.write_histogram(&histogram)?;

    if let Some((pathway_scores, pathway_predictions)) = score_pathway_representatives(
        inputs.args,
        inputs.progress,
        config,
        peak_count,
        inputs.records,
        &spectra,
        inputs.query_ids,
    )? {
        writers.write_pathway_scores(&pathway_scores)?;
        writers.write_pathway_predictions(&pathway_predictions)?;
    }

    writers.write_summary(&summary)?;
    Ok(distribution)
}

/// Compute one sorted score distribution from prepared spectra.
fn compute_score_distribution(
    inputs: &ScanInputs<'_>,
    config: &SimilarityConfig,
    config_name: &str,
    peak_count: usize,
    spectra: &[GenericSpectrum<f32>],
) -> Result<(ScoreDistribution, DistributionSummary)> {
    let hits = compute_neighbors(&SearchBatch {
        args: inputs.args,
        progress: inputs.progress,
        config,
        peak_count,
        records: inputs.records,
        spectra,
        query_ids: inputs.query_ids,
        reference_ids: inputs.reference_ids,
    })
    .with_context(|| format!("computing {config_name} neighbors for top {peak_count} peaks"))?;

    let mut scores = hits.into_iter().map(|hit| hit.score).collect::<Vec<_>>();
    scores.par_sort_by(f64::total_cmp);

    let summary = summarize_sorted_distribution(inputs.args, config, peak_count, &scores)?;
    let distribution = ScoreDistribution {
        peak_count,
        scores,
        mean: summary.mean,
    };
    Ok((distribution, summary))
}

/// Write regenerated artifacts for a cached score distribution.
fn write_cached_distribution_outputs(
    inputs: &ScanInputs<'_>,
    writers: &mut OutputWriters,
    config: &SimilarityConfig,
    distribution: &ScoreDistribution,
) -> Result<()> {
    let peak_count = distribution.peak_count;
    let summary =
        summarize_sorted_distribution(inputs.args, config, peak_count, &distribution.scores)?;
    let histogram =
        histogram_sorted_distribution(inputs.args, config, peak_count, &distribution.scores)?;
    writers.write_histogram(&histogram)?;

    if inputs.args.pathway_representatives_per_class > 0 {
        let spectra = prepare_spectra(
            inputs.progress,
            inputs.records,
            peak_count,
            inputs.args.mz_tolerance,
            !inputs.args.no_merge_close_peaks,
        )
        .with_context(|| {
            format!("preparing spectra for cached pathway scoring for top {peak_count} peaks")
        })?;
        if let Some((pathway_scores, pathway_predictions)) = score_pathway_representatives(
            inputs.args,
            inputs.progress,
            config,
            peak_count,
            inputs.records,
            &spectra,
            inputs.query_ids,
        )? {
            writers.write_pathway_scores(&pathway_scores)?;
            writers.write_pathway_predictions(&pathway_predictions)?;
        }
    }

    writers.write_summary(&summary)?;
    Ok(())
}

/// Write adjacent peak-count comparison rows for one similarity config.
fn write_adjacent_comparisons(
    inputs: &ScanInputs<'_>,
    writers: &mut OutputWriters,
    config: &SimilarityConfig,
    config_name: &str,
    distributions: &[ScoreDistribution],
) -> Result<()> {
    let adjacent_progress = inputs.progress.bar(
        u64::try_from(PEAK_COUNT_GRID_SIZE.saturating_sub(1)).unwrap_or(u64::MAX),
        format!("comparing adjacent distributions for {config_name}"),
    );
    let comparisons = (0..distributions.len().saturating_sub(1))
        .into_par_iter()
        .map(|index| {
            let comparison = compare_distributions(
                inputs.args,
                config,
                &distributions[index],
                &distributions[index + 1],
            );
            adjacent_progress.inc(1);
            comparison
        })
        .collect::<Result<Vec<_>>>()?;
    for comparison in comparisons {
        writers.write_adjacent_comparison(comparison);
    }
    adjacent_progress.finish();
    writers.flush_adjacent_comparisons()
}

/// Write full peak-count grid comparison rows for one similarity config.
fn write_grid_comparisons(
    inputs: &ScanInputs<'_>,
    writers: &mut OutputWriters,
    config: &SimilarityConfig,
    config_name: &str,
    distributions: &[ScoreDistribution],
) -> Result<()> {
    let grid_progress = inputs.progress.bar(
        u64::try_from(PEAK_COUNT_GRID_SIZE.saturating_mul(PEAK_COUNT_GRID_SIZE))
            .unwrap_or(u64::MAX),
        format!("comparing full distribution grid for {config_name}"),
    );
    let width = distributions.len();
    let comparisons = (0..width.saturating_mul(width))
        .into_par_iter()
        .map(|cell| {
            let row = cell / width;
            let column = cell % width;
            let comparison = if row == column {
                Ok(self_comparison(inputs.args, config, &distributions[row]))
            } else {
                compare_distributions(
                    inputs.args,
                    config,
                    &distributions[row],
                    &distributions[column],
                )
            };
            grid_progress.inc(1);
            comparison
        })
        .collect::<Result<Vec<_>>>()?;
    for comparison in comparisons {
        writers.write_grid_comparison(comparison);
    }
    grid_progress.finish();
    writers.flush_grid_comparisons()
}

#[cfg(test)]
/// Unit tests for resumable scan orchestration.
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    use anyhow::{Context, Result};
    use clap::Parser;
    use mass_spectrometry::prelude::GenericSpectrum;

    use crate::{
        checkpoint::{self, CheckpointBase},
        cli::{Cli, Commands, FinalizeScanArgs},
        data::load_records,
        model::{LoadedRecord, PEAK_COUNT_GRID_SIZE, ScoreDistribution},
        output,
        spectra::{select_query_ids, select_reference_ids},
    };

    use super::{
        ensure_pathway_labels_available, run_finalize_scan, run_scan, run_scan_shard,
        select_shard_assignment,
    };

    #[test]
    /// A valid cached distribution is reused while missing peak counts are computed.
    fn scan_reuses_valid_distribution_checkpoint() -> Result<()> {
        let root = temp_root("resume")?;
        let data_dir = root.join("data");
        let output_dir = root.join("out");
        fs::create_dir_all(&data_dir)?;
        fs::create_dir_all(&output_dir)?;

        let mut args = scan_args(&data_dir, &output_dir)?;
        args.validate()?;
        let mut records = load_records(&args)?;
        if let Some(max_spectra) = args.max_spectra {
            records.truncate(max_spectra.min(records.len()));
        }
        let query_ids = select_query_ids(records.len(), args.row_sample_size, args.seed);
        let reference_ids =
            select_reference_ids(records.len(), args.reference_sample_size, args.seed);
        let config = args
            .similarity_configs
            .first()
            .context("test scan must include one similarity config")?;
        let config_name = config.name();
        let checkpoint_base = CheckpointBase::new(&args, &records, &query_ids, &reference_ids);
        let fingerprint = checkpoint_base.fingerprint(&args, config, &config_name);
        let cached = ScoreDistribution {
            peak_count: 1,
            scores: vec![0.123_456, 0.654_321, 0.777_777],
            mean: 0.518_518,
        };
        checkpoint::store_distribution(
            &output_dir,
            args.dataset.as_str(),
            &config_name,
            config.metric_label(),
            &cached,
            &fingerprint,
        )?;

        run_scan(args)?;

        let retained = checkpoint::load_distribution(
            &output_dir,
            "synthetic-smoke",
            &config_name,
            "cosine",
            1,
            &fingerprint,
        );
        assert_eq!(retained, Some(cached));
        assert!(
            checkpoint::checkpoint_path(&output_dir, &config_name, 2).exists(),
            "missing peak counts should be computed and checkpointed"
        );

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    /// A cached distribution shard still backfills a missing pathway shard.
    fn scan_shard_with_cached_distribution_writes_missing_pathway_shard() -> Result<()> {
        let root = temp_root("cached-shard-pathways")?;
        let data_dir = root.join("data");
        let output_dir = root.join("out");
        fs::create_dir_all(&data_dir)?;
        fs::create_dir_all(&output_dir)?;

        let mut args = explicit_scan_shard_args(&data_dir, &output_dir, "1", false)?;
        args.scan.pathway_representatives_per_class = 1;
        args.scan.validate()?;
        let config = args
            .scan
            .similarity_configs
            .first()
            .context("test scan must include one similarity config")?;
        let config_name = config.name();
        let checkpoint_base = checkpoint_base_for_args(&args.scan)?;
        let fingerprint = checkpoint_base.fingerprint(&args.scan, config, &config_name);
        let cached = test_distribution(1);

        checkpoint::store_distribution(
            &output_dir,
            args.scan.dataset.as_str(),
            &config_name,
            config.metric_label(),
            &cached,
            &fingerprint,
        )?;

        assert!(
            !output::pathway_shard_exists(&output_dir, &config_name, 1),
            "pathway shard fixture should start missing"
        );

        run_scan_shard(args)?;

        let retained = checkpoint::load_distribution(
            &output_dir,
            "synthetic-smoke",
            &config_name,
            "cosine",
            1,
            &fingerprint,
        );
        assert_eq!(retained, Some(cached));
        assert!(
            output::pathway_shard_exists(&output_dir, &config_name, 1),
            "cached shard should still write missing pathway artifacts"
        );

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    /// Finalization fails before aggregation when pathway shards are missing.
    fn finalize_scan_requires_pathway_shards_when_requested() -> Result<()> {
        let root = temp_root("finalize-pathway-shards")?;
        let data_dir = root.join("data");
        let output_dir = root.join("out");
        fs::create_dir_all(&data_dir)?;
        fs::create_dir_all(&output_dir)?;

        let mut scan = scan_args(&data_dir, &output_dir)?;
        scan.pathway_representatives_per_class = 1;
        scan.validate()?;
        let config = scan
            .similarity_configs
            .first()
            .context("test scan must include one similarity config")?;
        let config_name = config.name();
        let checkpoint_base = checkpoint_base_for_args(&scan)?;
        let fingerprint = checkpoint_base.fingerprint(&scan, config, &config_name);

        for peak_count in 1..=PEAK_COUNT_GRID_SIZE {
            checkpoint::store_distribution(
                &output_dir,
                scan.dataset.as_str(),
                &config_name,
                config.metric_label(),
                &test_distribution(peak_count),
                &fingerprint,
            )?;
        }

        let Err(error) = run_finalize_scan(FinalizeScanArgs { scan }) else {
            anyhow::bail!("finalizing without pathway shards should fail");
        };
        assert!(
            error
                .to_string()
                .contains("missing pathway prediction shards"),
            "unexpected error: {error}"
        );

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    /// Pathway scoring fails before a long scan when no pathway labels are loaded.
    fn pathway_scoring_requires_loaded_pathway_labels() -> Result<()> {
        let root = temp_root("pathway-labels")?;
        let data_dir = root.join("data");
        let output_dir = root.join("out");
        fs::create_dir_all(&data_dir)?;

        let mut args = scan_args(&data_dir, &output_dir)?;
        args.pathway_representatives_per_class = 5;
        let records = vec![LoadedRecord {
            id: "unlabeled".to_string(),
            npc_pathway: None,
            spectrum: GenericSpectrum::<f32>::try_with_capacity(100.0, 0)?,
        }];

        let Err(error) = ensure_pathway_labels_available(&args, &records) else {
            anyhow::bail!("pathway scoring without labels should fail");
        };
        assert!(
            error.to_string().contains("NPC_PATHWAYS"),
            "unexpected error: {error}"
        );

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    /// Scan-shard selector arguments reject ambiguous and out-of-range requests.
    fn scan_shard_selector_rejects_ambiguous_and_out_of_range_requests() -> Result<()> {
        let root = temp_root("shard-selector-errors")?;
        let data_dir = root.join("data");
        let output_dir = root.join("out");
        fs::create_dir_all(&data_dir)?;

        let mut both_selectors = explicit_scan_shard_args(&data_dir, &output_dir, "1", false)?;
        both_selectors.shard_index = Some(0);
        let Err(error) = select_shard_assignment(&both_selectors) else {
            anyhow::bail!("scan-shard with both selectors should fail");
        };
        assert!(
            error.to_string().contains("not both"),
            "unexpected error: {error}"
        );

        let mut neither_selector = explicit_scan_shard_args(&data_dir, &output_dir, "1", false)?;
        neither_selector.peak_count = None;
        let Err(error) = select_shard_assignment(&neither_selector) else {
            anyhow::bail!("scan-shard without a selector should fail");
        };
        assert!(
            error.to_string().contains("requires either"),
            "unexpected error: {error}"
        );

        let out_of_range = scan_shard_args(&data_dir, &output_dir, "2304")?;
        let Err(error) = select_shard_assignment(&out_of_range) else {
            anyhow::bail!("out-of-range shard index should fail");
        };
        assert!(
            error.to_string().contains("outside the selected grid"),
            "unexpected error: {error}"
        );

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    /// Shard indexes map row-major over configs and one-based peak counts.
    fn shard_index_mapping_matches_config_peak_grid() -> Result<()> {
        let root = temp_root("shard-index")?;
        let data_dir = root.join("data");
        let output_dir = root.join("out");
        fs::create_dir_all(&data_dir)?;

        let first = select_shard_assignment(&scan_shard_args(&data_dir, &output_dir, "0")?)?;
        assert_eq!(first.config.name(), "cosine_mz0.000_int1.000");
        assert_eq!(first.config_index, 0);
        assert_eq!(first.peak_count, 1);

        let last_first_config =
            select_shard_assignment(&scan_shard_args(&data_dir, &output_dir, "127")?)?;
        assert_eq!(last_first_config.config.name(), "cosine_mz0.000_int1.000");
        assert_eq!(last_first_config.config_index, 0);
        assert_eq!(last_first_config.peak_count, 128);

        let first_second_config =
            select_shard_assignment(&scan_shard_args(&data_dir, &output_dir, "128")?)?;
        assert_eq!(
            first_second_config.config.name(),
            "modified_cosine_mz0.000_int1.000"
        );
        assert_eq!(first_second_config.config_index, 1);
        assert_eq!(first_second_config.peak_count, 1);

        let last = select_shard_assignment(&scan_shard_args(&data_dir, &output_dir, "2303")?)?;
        assert_eq!(
            last.config.name(),
            "modified_entropy_mz0.000_int1.000_weightedfalse"
        );
        assert_eq!(last.config_index, 17);
        assert_eq!(last.peak_count, 128);

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    /// Explicit scan shards validate peak count and config cardinality.
    fn explicit_scan_shard_requires_one_config_and_valid_peak_count() -> Result<()> {
        let root = temp_root("explicit-shard")?;
        let data_dir = root.join("data");
        let output_dir = root.join("out");
        fs::create_dir_all(&data_dir)?;

        let invalid_peak = explicit_scan_shard_args(&data_dir, &output_dir, "0", false)?;
        let Err(error) = select_shard_assignment(&invalid_peak) else {
            anyhow::bail!("invalid peak count should fail");
        };
        assert!(
            error.to_string().contains("--peak-count must be in"),
            "unexpected error: {error}"
        );

        let multiple_configs = explicit_scan_shard_args(&data_dir, &output_dir, "1", true)?;
        let Err(error) = select_shard_assignment(&multiple_configs) else {
            anyhow::bail!("multiple configs should fail");
        };
        assert!(
            error
                .to_string()
                .contains("exactly one --similarity-config"),
            "unexpected error: {error}"
        );

        fs::remove_dir_all(root)?;
        Ok(())
    }

    /// Build the checkpoint fingerprint base for a synthetic scan fixture.
    fn checkpoint_base_for_args(args: &crate::cli::ScanArgs) -> Result<CheckpointBase> {
        let mut records = load_records(args)?;
        if let Some(max_spectra) = args.max_spectra {
            records.truncate(max_spectra.min(records.len()));
        }
        let query_ids = select_query_ids(records.len(), args.row_sample_size, args.seed);
        let reference_ids =
            select_reference_ids(records.len(), args.reference_sample_size, args.seed);
        Ok(CheckpointBase::new(
            args,
            &records,
            &query_ids,
            &reference_ids,
        ))
    }

    /// Return a tiny sorted distribution fixture for one retained-peak count.
    fn test_distribution(peak_count: usize) -> ScoreDistribution {
        let peak_count_f64 = f64::from(u32::try_from(peak_count).unwrap_or(u32::MAX));
        let scores = vec![0.25, 0.5, peak_count_f64.mul_add(0.000_001, 0.75)];
        let mean =
            scores.iter().sum::<f64>() / f64::from(u32::try_from(scores.len()).unwrap_or(u32::MAX));
        ScoreDistribution {
            peak_count,
            scores,
            mean,
        }
    }

    /// Parse the synthetic scan arguments used by the resume test.
    fn scan_args(data_dir: &Path, output_dir: &Path) -> Result<crate::cli::ScanArgs> {
        let cli = Cli::try_parse_from([
            "spectral-similarities-by-peaks",
            "scan",
            "--dataset",
            "synthetic-smoke",
            "--data-dir",
            data_dir
                .to_str()
                .context("temporary data path is not valid UTF-8")?,
            "--output-dir",
            output_dir
                .to_str()
                .context("temporary output path is not valid UTF-8")?,
            "--similarity-config",
            "cosine:0.0:1.0",
            "--neighbors",
            "2",
            "--mz-tolerance",
            "0.05",
            "--histogram-bins",
            "4",
            "--row-sample-size",
            "4",
            "--reference-sample-size",
            "6",
            "--max-spectra",
            "8",
            "--seed",
            "99",
        ])?;
        match cli.command {
            Commands::Scan(args) => Ok(args),
            Commands::Prefetch(_)
            | Commands::ScanShard(_)
            | Commands::FinalizeScan(_)
            | Commands::FinalizeShard(_)
            | Commands::FinalizeMerge(_)
            | Commands::RenderHeatmaps(_)
            | Commands::RenderPathwayArtifacts(_)
            | Commands::ComputePathwayDiscriminability(_)
            | Commands::ComputeConfigDiversity(_)
            | Commands::ReEncodeParquets(_) => {
                anyhow::bail!("expected scan command")
            }
        }
    }

    /// Parse scan-shard arguments using the default similarity config grid.
    fn scan_shard_args(
        data_dir: &Path,
        output_dir: &Path,
        shard_index: &str,
    ) -> Result<crate::cli::ScanShardArgs> {
        let cli = Cli::try_parse_from([
            "spectral-similarities-by-peaks",
            "scan-shard",
            "--dataset",
            "synthetic-smoke",
            "--data-dir",
            data_dir
                .to_str()
                .context("temporary data path is not valid UTF-8")?,
            "--output-dir",
            output_dir
                .to_str()
                .context("temporary output path is not valid UTF-8")?,
            "--shard-index",
            shard_index,
        ])?;
        match cli.command {
            Commands::ScanShard(args) => Ok(args),
            Commands::Prefetch(_)
            | Commands::Scan(_)
            | Commands::FinalizeScan(_)
            | Commands::FinalizeShard(_)
            | Commands::FinalizeMerge(_)
            | Commands::RenderHeatmaps(_)
            | Commands::RenderPathwayArtifacts(_)
            | Commands::ComputePathwayDiscriminability(_)
            | Commands::ComputeConfigDiversity(_)
            | Commands::ReEncodeParquets(_) => {
                anyhow::bail!("expected scan-shard command")
            }
        }
    }

    /// Parse explicit scan-shard arguments for validation tests.
    fn explicit_scan_shard_args(
        data_dir: &Path,
        output_dir: &Path,
        peak_count: &str,
        include_second_config: bool,
    ) -> Result<crate::cli::ScanShardArgs> {
        let mut args = vec![
            "spectral-similarities-by-peaks",
            "scan-shard",
            "--dataset",
            "synthetic-smoke",
            "--data-dir",
            data_dir
                .to_str()
                .context("temporary data path is not valid UTF-8")?,
            "--output-dir",
            output_dir
                .to_str()
                .context("temporary output path is not valid UTF-8")?,
            "--similarity-config",
            "cosine:0.0:1.0",
            "--peak-count",
            peak_count,
        ];
        if include_second_config {
            args.extend(["--similarity-config", "entropy:0.0:1.0:false"]);
        }
        let cli = Cli::try_parse_from(args)?;
        match cli.command {
            Commands::ScanShard(args) => Ok(args),
            Commands::Prefetch(_)
            | Commands::Scan(_)
            | Commands::FinalizeScan(_)
            | Commands::FinalizeShard(_)
            | Commands::FinalizeMerge(_)
            | Commands::RenderHeatmaps(_)
            | Commands::RenderPathwayArtifacts(_)
            | Commands::ComputePathwayDiscriminability(_)
            | Commands::ComputeConfigDiversity(_)
            | Commands::ReEncodeParquets(_) => {
                anyhow::bail!("expected scan-shard command")
            }
        }
    }

    /// Return a unique temporary directory for one scan test.
    fn temp_root(label: &str) -> Result<PathBuf> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?;
        Ok(std::env::temp_dir().join(format!(
            "spectral-similarities-run-{label}-{}-{}",
            std::process::id(),
            timestamp.as_nanos()
        )))
    }
}
